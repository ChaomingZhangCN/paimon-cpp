/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import org.apache.paimon.catalog.Catalog;
import org.apache.paimon.catalog.CatalogContext;
import org.apache.paimon.catalog.CatalogFactory;
import org.apache.paimon.catalog.Identifier;
import org.apache.paimon.data.BinaryString;
import org.apache.paimon.data.BinaryVector;
import org.apache.paimon.data.GenericRow;
import org.apache.paimon.data.InternalRow;
import org.apache.paimon.io.DataFileMeta;
import org.apache.paimon.manifest.ManifestEntry;
import org.apache.paimon.options.Options;
import org.apache.paimon.schema.Schema;
import org.apache.paimon.table.FileStoreTable;
import org.apache.paimon.table.Table;
import org.apache.paimon.table.sink.BatchTableCommit;
import org.apache.paimon.table.sink.BatchTableWrite;
import org.apache.paimon.table.sink.BatchWriteBuilder;
import org.apache.paimon.table.source.ReadBuilder;
import org.apache.paimon.types.DataTypes;
import org.apache.paimon.utils.CloseableIterator;

import java.util.Arrays;
import java.util.BitSet;
import java.util.Map;
import java.util.TreeMap;

/** Generates one Java main file with independently rolled dedicated VECTOR files. */
public class GenerateDedicatedVectorRollingFixture {
    private static final int ROW_COUNT = 3001;

    public static void main(String[] args) throws Exception {
        if (args.length != 1) {
            throw new IllegalArgumentException("Expected an empty warehouse directory argument");
        }
        Options options = new Options();
        options.set("warehouse", args[0]);
        Identifier identifier =
                Identifier.create("dedicated_vector_java_rolling", "dedicated_vector_java_rolling");
        try (Catalog catalog = CatalogFactory.createCatalog(CatalogContext.create(options))) {
            catalog.createDatabase(identifier.getDatabaseName(), false);
            Schema schema =
                    Schema.newBuilder()
                            .column("id", DataTypes.INT())
                            .column("embedding", DataTypes.VECTOR(3, DataTypes.FLOAT()))
                            .column("tag", DataTypes.STRING())
                            .option("file.format", "parquet")
                            .option("vector.file.format", "vortex")
                            .option("bucket", "-1")
                            .option("row-tracking.enabled", "true")
                            .option("data-evolution.enabled", "true")
                            .option("write-only", "true")
                            .option("target-file-size", "128 mb")
                            .option("vector.target-file-size", "1 kb")
                            .option("write.batch-size", "100")
                            .option("read.batch-size", "127")
                            .build();
            catalog.createTable(identifier, schema, false);
            Table table = catalog.getTable(identifier);
            BatchWriteBuilder builder = table.newBatchWriteBuilder();
            try (BatchTableWrite write = builder.newWrite();
                    BatchTableCommit commit = builder.newCommit()) {
                for (int id = 0; id < ROW_COUNT; id++) {
                    write.write(
                            GenericRow.of(
                                    id,
                                    isNullVector(id)
                                            ? null
                                            : BinaryVector.fromPrimitiveArray(vector(id)),
                                    BinaryString.fromString("row-" + id)));
                }
                commit.commit(write.prepareCommit());
            }

            ReadBuilder read = table.newReadBuilder();
            BitSet seen = new BitSet(ROW_COUNT);
            int readCount = 0;
            try (CloseableIterator<InternalRow> rows =
                    read.newRead().createReader(read.newScan().plan()).toCloseableIterator()) {
                while (rows.hasNext()) {
                    InternalRow row = rows.next();
                    int id = row.getInt(0);
                    if (id < 0 || id >= ROW_COUNT || seen.get(id)) {
                        throw new AssertionError("Unexpected or duplicate row id: " + id);
                    }
                    seen.set(id);
                    readCount++;
                    if (!row.getString(2).toString().equals("row-" + id)) {
                        throw new AssertionError("Unexpected tag for row " + id);
                    }
                    if (row.isNullAt(1) != isNullVector(id)
                            || (!isNullVector(id)
                                    && !Arrays.equals(
                                            row.getVector(1).toFloatArray(), vector(id)))) {
                        throw new AssertionError("Unexpected vector for row " + id);
                    }
                }
            }
            if (readCount != ROW_COUNT) {
                throw new AssertionError("Unexpected Java row count: " + readCount);
            }
            System.out.println("Java readback verified: " + readCount + " rows");

            int mainFiles = 0;
            Map<Long, Long> vectorRanges = new TreeMap<>();
            for (ManifestEntry entry : ((FileStoreTable) table).store().newScan().plan().files()) {
                DataFileMeta file = entry.file();
                if (file.fileName().endsWith(".parquet")) {
                    mainFiles++;
                    if (file.nonNullFirstRowId() != 0
                            || file.rowCount() != ROW_COUNT
                            || !file.writeCols().equals(Arrays.asList("id", "tag"))) {
                        throw new AssertionError("Unexpected main file metadata: " + file);
                    }
                } else if (file.fileName().endsWith(".vector.vortex")) {
                    if (!file.writeCols().equals(Arrays.asList("embedding"))
                            || vectorRanges.put(file.nonNullFirstRowId(), file.rowCount())
                                    != null) {
                        throw new AssertionError("Unexpected VECTOR file metadata: " + file);
                    }
                } else {
                    throw new AssertionError("Unexpected data file: " + file);
                }
                System.out.println(
                        file.fileName()
                                + " firstRowId="
                                + file.firstRowId()
                                + " rowCount="
                                + file.rowCount()
                                + " fileSize="
                                + file.fileSize()
                                + " writeCols="
                                + file.writeCols());
            }
            Map<Long, Long> expectedVectorRanges = new TreeMap<>();
            expectedVectorRanges.put(0L, 1000L);
            expectedVectorRanges.put(1000L, 1000L);
            expectedVectorRanges.put(2000L, 1000L);
            expectedVectorRanges.put(3000L, 1L);
            if (mainFiles != 1 || !vectorRanges.equals(expectedVectorRanges)) {
                throw new AssertionError(
                        "Expected one main and four independently rolled VECTOR files: "
                                + mainFiles
                                + " main, VECTOR ranges "
                                + vectorRanges);
            }
        }
    }

    private static boolean isNullVector(int id) {
        return id % 1000 == 0 || id % 1000 == 999;
    }

    private static float[] vector(int id) {
        return new float[] {id, id + 1, id + 2};
    }
}
