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
#include <vector>

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

    void ReadFixtureAndCheck(
        const std::string& file_name, arrow::Type::type expected_file_vector_type,
        int32_t vector_length, const std::vector<int32_t>& expected_ids,
        const std::vector<std::optional<std::vector<float>>>& expected_vectors) {
        std::string file_path =
            paimon::test::GetDataDir() + "/parquet/vector_compatibility/" + file_name;
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
        std::shared_ptr<arrow::Field> file_vector_field = file_type->GetFieldByName("embedding");
        ASSERT_TRUE(file_vector_field);
        ASSERT_EQ(file_vector_field->type()->id(), expected_file_vector_type);
        std::shared_ptr<arrow::Field> file_id_field = file_type->GetFieldByName("id");
        ASSERT_TRUE(file_id_field);

        auto vector_type = arrow::fixed_size_list(
            arrow::field("element", arrow::float32(), /*nullable=*/false), vector_length);
        auto logical_schema =
            arrow::schema({file_id_field, file_vector_field->WithType(vector_type)});
        std::unique_ptr<FileBatchReader> vector_reader =
            std::make_unique<VectorFileBatchReader>(std::move(reader), pool_);
        auto c_read_schema = std::make_unique<ArrowSchema>();
        ASSERT_TRUE(arrow::ExportSchema(*logical_schema, c_read_schema.get()).ok());
        ASSERT_OK(vector_reader->SetReadSchema(c_read_schema.get(), /*predicate=*/nullptr,
                                               /*selection_bitmap=*/std::nullopt));
        ASSERT_OK_AND_ASSIGN(std::shared_ptr<arrow::ChunkedArray> actual,
                             paimon::test::ReadResultCollector::CollectResult(vector_reader.get()));
        ASSERT_EQ(actual->num_chunks(), 1);
        ASSERT_EQ(actual->type()->id(), arrow::Type::STRUCT);
        auto struct_array = checked_pointer_cast<arrow::StructArray>(actual->chunk(0));
        std::shared_ptr<arrow::Array> id_field = struct_array->GetFieldByName("id");
        std::shared_ptr<arrow::Array> vector_field = struct_array->GetFieldByName("embedding");
        ASSERT_TRUE(id_field);
        ASSERT_TRUE(vector_field);
        ASSERT_EQ(id_field->type_id(), arrow::Type::INT32);
        ASSERT_EQ(vector_field->type_id(), arrow::Type::FIXED_SIZE_LIST);
        auto vector_array = checked_pointer_cast<arrow::FixedSizeListArray>(vector_field);
        ASSERT_EQ(id_field->length(), static_cast<int64_t>(expected_ids.size()));
        ASSERT_EQ(vector_array->length(), static_cast<int64_t>(expected_vectors.size()));
        const int32_t* id_values = id_field->data()->GetValues<int32_t>(1);
        for (int64_t i = 0; i < id_field->length(); ++i) {
            ASSERT_FALSE(id_field->IsNull(i));
            ASSERT_EQ(id_values[id_field->offset() + i], expected_ids[i]);
            if (!expected_vectors[i]) {
                ASSERT_TRUE(vector_array->IsNull(i));
                continue;
            }
            ASSERT_FALSE(vector_array->IsNull(i));
            ASSERT_EQ(vector_array->value_length(i),
                      static_cast<int64_t>(expected_vectors[i]->size()));
            std::shared_ptr<arrow::Array> values = vector_array->value_slice(i);
            const float* vector_values = values->data()->GetValues<float>(1);
            for (int64_t j = 0; j < vector_array->value_length(i); ++j) {
                ASSERT_FLOAT_EQ(vector_values[j], expected_vectors[i].value()[j]);
            }
        }
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

TEST_F(ParquetVectorIoTest, ReadJavaFixture) {
    ReadFixtureAndCheck(
        "java_vector.parquet", arrow::Type::LIST, /*vector_length=*/2,
        /*expected_ids=*/{0, 1, 2, 3, 4},
        /*expected_vectors=*/
        {{{0.0f, 0.0f}}, {{1.0f, 0.0f}}, {{2.0f, 0.0f}}, {{3.0f, 0.0f}}, {{4.0f, 0.0f}}});
}

TEST_F(ParquetVectorIoTest, ReadRustFixture) {
    ReadFixtureAndCheck("rust_vector.parquet", arrow::Type::FIXED_SIZE_LIST,
                        /*vector_length=*/3, /*expected_ids=*/{1, 2, 3},
                        /*expected_vectors=*/
                        {{{1.0f, 2.0f, 3.0f}}, {{7.0f, 8.0f, 9.0f}}, {{4.0f, 5.0f, 6.0f}}});
}

}  // namespace paimon::parquet::test
