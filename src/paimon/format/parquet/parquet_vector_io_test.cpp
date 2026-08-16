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

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "arrow/api.h"
#include "arrow/c/abi.h"
#include "arrow/c/bridge.h"
#include "arrow/ipc/json_simple.h"
#include "gtest/gtest.h"
#include "paimon/common/utils/arrow/arrow_input_stream_adapter.h"
#include "paimon/common/utils/arrow/mem_utils.h"
#include "paimon/common/utils/checked_cast.h"
#include "paimon/core/io/vector_file_batch_reader.h"
#include "paimon/format/parquet/parquet_file_batch_reader.h"
#include "paimon/format/parquet/parquet_format_defs.h"
#include "paimon/format/parquet/parquet_format_writer.h"
#include "paimon/fs/file_system.h"
#include "paimon/memory/memory_pool.h"
#include "paimon/testing/utils/read_result_collector.h"
#include "paimon/testing/utils/testharness.h"
#include "parquet/properties.h"

namespace paimon::parquet::test {

class ParquetVectorIoTest : public ::testing::Test {
 public:
    void SetUp() override {
        pool_ = GetDefaultPool();
        arrow_pool_ = GetArrowPool(pool_);
        dir_ = paimon::test::UniqueTestDirectory::Create();
        ASSERT_TRUE(dir_);
        fs_ = dir_->GetFileSystem();
    }

    void WriteAndCheck(const std::string& file_name,
                       const std::shared_ptr<arrow::StructType>& write_type,
                       const std::shared_ptr<arrow::StructType>& read_type,
                       const std::string& json) {
        arrow::Result<std::shared_ptr<arrow::Array>> write_array_result =
            arrow::ipc::internal::json::ArrayFromJSON(write_type, json);
        ASSERT_TRUE(write_array_result.ok()) << write_array_result.status().ToString();
        std::shared_ptr<arrow::Array> write_array = std::move(write_array_result).ValueOrDie();
        auto c_array = std::make_unique<ArrowArray>();
        ASSERT_TRUE(arrow::ExportArray(*write_array, c_array.get()).ok());

        std::string file_path = dir_->Str() + "/" + file_name;
        ASSERT_OK_AND_ASSIGN(std::shared_ptr<OutputStream> out,
                             fs_->Create(file_path, /*overwrite=*/false));
        ::parquet::WriterProperties::Builder properties_builder;
        ASSERT_OK_AND_ASSIGN(
            std::unique_ptr<ParquetFormatWriter> writer,
            ParquetFormatWriter::Create(out, arrow::schema(write_type->fields()),
                                        properties_builder.build(),
                                        DEFAULT_PARQUET_WRITER_MAX_MEMORY_USE, arrow_pool_));
        ASSERT_OK(writer->AddBatch(c_array.get()));
        ASSERT_OK(writer->Finish());
        ASSERT_OK(out->Close());

        ASSERT_OK_AND_ASSIGN(std::shared_ptr<InputStream> in, fs_->Open(file_path));
        ASSERT_OK_AND_ASSIGN(int64_t length, in->Length());
        auto in_stream = std::make_shared<ArrowInputStreamAdapter>(in, length, arrow_pool_);
        ASSERT_OK_AND_ASSIGN(
            std::unique_ptr<ParquetFileBatchReader> reader,
            ParquetFileBatchReader::Create(std::move(in_stream), /*options=*/{},
                                           /*batch_size=*/10, /*file_metadata=*/nullptr,
                                           /*storage_read_bytes=*/nullptr, arrow_pool_));
        ASSERT_OK_AND_ASSIGN(std::unique_ptr<ArrowSchema> c_file_schema, reader->GetFileSchema());
        arrow::Result<std::shared_ptr<arrow::DataType>> file_type_result =
            arrow::ImportType(c_file_schema.get());
        ASSERT_TRUE(file_type_result.ok()) << file_type_result.status().ToString();
        auto file_type =
            checked_pointer_cast<arrow::StructType>(std::move(file_type_result).ValueOrDie());
        std::shared_ptr<arrow::DataType> physical_value_type = file_type->field(1)->type();
        if (physical_value_type->id() == arrow::Type::STRUCT) {
            physical_value_type = physical_value_type->field(0)->type();
        }
        ASSERT_EQ(physical_value_type->id(), arrow::Type::LIST);

        std::unique_ptr<FileBatchReader> vector_reader =
            std::make_unique<VectorFileBatchReader>(std::move(reader), pool_);
        auto c_schema = std::make_unique<ArrowSchema>();
        ASSERT_TRUE(arrow::ExportSchema(*arrow::schema(read_type->fields()), c_schema.get()).ok());
        ASSERT_OK(vector_reader->SetReadSchema(c_schema.get(), /*predicate=*/nullptr,
                                               /*selection_bitmap=*/std::nullopt));
        ASSERT_OK_AND_ASSIGN(std::shared_ptr<arrow::ChunkedArray> actual,
                             paimon::test::ReadResultCollector::CollectResult(vector_reader.get()));

        arrow::Result<std::shared_ptr<arrow::Array>> expected_result =
            arrow::ipc::internal::json::ArrayFromJSON(read_type, json);
        ASSERT_TRUE(expected_result.ok()) << expected_result.status().ToString();
        std::shared_ptr<arrow::Array> expected = std::move(expected_result).ValueOrDie();
        ASSERT_TRUE(std::make_shared<arrow::ChunkedArray>(expected)->Equals(actual))
            << actual->ToString();
    }

 private:
    std::shared_ptr<MemoryPool> pool_;
    std::shared_ptr<arrow::MemoryPool> arrow_pool_;
    std::shared_ptr<FileSystem> fs_;
    std::unique_ptr<paimon::test::UniqueTestDirectory> dir_;
};

TEST_F(ParquetVectorIoTest, WriteAndReadVector) {
    auto vector_type =
        arrow::fixed_size_list(arrow::field("item", arrow::float32(), /*nullable=*/false), 3);
    auto struct_type = checked_pointer_cast<arrow::StructType>(arrow::struct_(
        {arrow::field("id", arrow::int32()), arrow::field("embedding", vector_type)}));
    WriteAndCheck("vector.parquet", struct_type, struct_type,
                  R"([[1, [1.0, 2.0, 3.0]], [2, null], [3, [4.0, 5.0, 6.0]]])");
}

TEST_F(ParquetVectorIoTest, ReadOrdinaryParquetListAsVector) {
    auto physical_type = checked_pointer_cast<arrow::StructType>(
        arrow::struct_({arrow::field("id", arrow::int32()),
                        arrow::field("embedding", arrow::list(arrow::float32()))}));
    auto logical_type = checked_pointer_cast<arrow::StructType>(arrow::struct_({
        arrow::field("id", arrow::int32()),
        arrow::field("embedding", arrow::fixed_size_list(arrow::float32(), 3)),
    }));
    WriteAndCheck("list.parquet", physical_type, logical_type,
                  R"([[1, [1.0, 2.0, 3.0]], [2, null], [3, [4.0, 5.0, 6.0]]])");
}

TEST_F(ParquetVectorIoTest, WriteAndReadNestedDoubleVector) {
    auto vector_type =
        arrow::fixed_size_list(arrow::field("item", arrow::float64(), /*nullable=*/false), 2);
    auto struct_type = checked_pointer_cast<arrow::StructType>(arrow::struct_({
        arrow::field("id", arrow::int32()),
        arrow::field("payload", arrow::struct_({arrow::field("embedding", vector_type),
                                                arrow::field("history", arrow::list(vector_type)),
                                                arrow::field("by_name", arrow::map(arrow::utf8(),
                                                                                   vector_type))})),
    }));
    WriteAndCheck("nested-vector.parquet", struct_type, struct_type,
                  R"([[1, [[1.0, 2.0], [[3.0, 4.0], null], [["a", [5.0, 6.0]]]]],
                       [2, [null, null, [["b", null]]]]])");
}

}  // namespace paimon::parquet::test
