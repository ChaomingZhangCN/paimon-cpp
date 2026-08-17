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

#include "paimon/core/io/vector_file_batch_reader.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "arrow/api.h"
#include "arrow/c/bridge.h"
#include "arrow/ipc/json_simple.h"
#include "gtest/gtest.h"
#include "paimon/common/utils/checked_cast.h"
#include "paimon/memory/memory_pool.h"
#include "paimon/testing/mock/mock_file_batch_reader.h"
#include "paimon/testing/utils/testharness.h"

namespace paimon::test {
namespace {

std::shared_ptr<arrow::StructType> AsStructType(const std::shared_ptr<arrow::DataType>& type) {
    return checked_pointer_cast<arrow::StructType>(type);
}

}  // namespace

TEST(VectorFileBatchReaderTest, ConvertSchemaAndNextBatch) {
    auto physical_type = AsStructType(arrow::struct_({
        arrow::field("id", arrow::int32()),
        arrow::field("embedding", arrow::list(arrow::float32())),
    }));
    auto logical_type = AsStructType(arrow::struct_({
        arrow::field("id", arrow::int32()),
        arrow::field("embedding", arrow::fixed_size_list(arrow::float32(), 3)),
    }));
    const std::string json = R"([
        [1, [1.0, 2.0, 3.0]],
        [2, null],
        [3, [4.0, 5.0, 6.0]]
    ])";
    auto physical_array =
        arrow::ipc::internal::json::ArrayFromJSON(physical_type, json).ValueOrDie();
    auto mock_reader =
        std::make_unique<MockFileBatchReader>(physical_array, physical_type, /*batch_size=*/10);
    mock_reader->EnableRandomizeBatchSize(false);
    MockFileBatchReader* inner_reader = mock_reader.get();
    VectorFileBatchReader reader(std::move(mock_reader), GetDefaultPool());

    ASSERT_TRUE(VectorFileBatchReader::ContainsVector(arrow::schema(logical_type->fields())));
    ASSERT_FALSE(VectorFileBatchReader::ContainsVector(arrow::schema(physical_type->fields())));
    ArrowSchema c_read_schema;
    ASSERT_TRUE(arrow::ExportSchema(*arrow::schema(logical_type->fields()), &c_read_schema).ok());
    ASSERT_OK(reader.SetReadSchema(&c_read_schema, /*predicate=*/nullptr,
                                   /*selection_bitmap=*/std::nullopt));
    ASSERT_EQ(inner_reader->read_schema_->field(1)->type()->id(), arrow::Type::LIST);

    ASSERT_OK_AND_ASSIGN(BatchReader::ReadBatch batch, reader.NextBatch());
    arrow::Result<std::shared_ptr<arrow::Array>> actual_result =
        arrow::ImportArray(batch.first.get(), batch.second.get());
    ASSERT_TRUE(actual_result.ok()) << actual_result.status().ToString();
    std::shared_ptr<arrow::Array> actual = std::move(actual_result).ValueOrDie();
    auto expected = arrow::ipc::internal::json::ArrayFromJSON(logical_type, json).ValueOrDie();
    ASSERT_TRUE(expected->Equals(actual)) << actual->ToString();
    ASSERT_OK_AND_ASSIGN(batch, reader.NextBatch());
    ASSERT_TRUE(BatchReader::IsEofBatch(batch));
}

TEST(VectorFileBatchReaderTest, KeepFixedSizeListFileSchema) {
    auto logical_type = AsStructType(arrow::struct_({
        arrow::field("id", arrow::int32()),
        arrow::field("embedding", arrow::fixed_size_list(arrow::float32(), 3)),
    }));
    const std::string json = R"([
        [1, [1.0, 2.0, 3.0]],
        [2, null],
        [3, [4.0, 5.0, 6.0]]
    ])";
    auto logical_array = arrow::ipc::internal::json::ArrayFromJSON(logical_type, json).ValueOrDie();
    auto mock_reader =
        std::make_unique<MockFileBatchReader>(logical_array, logical_type, /*batch_size=*/10);
    mock_reader->EnableRandomizeBatchSize(false);
    MockFileBatchReader* inner_reader = mock_reader.get();
    VectorFileBatchReader reader(std::move(mock_reader), GetDefaultPool());

    ArrowSchema c_read_schema;
    ASSERT_TRUE(arrow::ExportSchema(*arrow::schema(logical_type->fields()), &c_read_schema).ok());
    ASSERT_OK(reader.SetReadSchema(&c_read_schema, /*predicate=*/nullptr,
                                   /*selection_bitmap=*/std::nullopt));
    ASSERT_EQ(inner_reader->read_schema_->field(1)->type()->id(), arrow::Type::FIXED_SIZE_LIST);

    ASSERT_OK_AND_ASSIGN(BatchReader::ReadBatch batch, reader.NextBatch());
    arrow::Result<std::shared_ptr<arrow::Array>> actual_result =
        arrow::ImportArray(batch.first.get(), batch.second.get());
    ASSERT_TRUE(actual_result.ok()) << actual_result.status().ToString();
    ASSERT_TRUE(logical_array->Equals(std::move(actual_result).ValueOrDie()));
}

TEST(VectorFileBatchReaderTest, ConvertNestedVectorsWithBitmap) {
    auto logical_vector =
        arrow::fixed_size_list(arrow::field("item", arrow::float64(), /*nullable=*/false), 2);
    auto physical_vector = arrow::list(arrow::field("item", arrow::float64(), /*nullable=*/false));
    auto logical_type = AsStructType(arrow::struct_({
        arrow::field("vectors", arrow::list(logical_vector)),
        arrow::field("by_name", arrow::map(arrow::utf8(), logical_vector)),
    }));
    auto physical_type = AsStructType(arrow::struct_({
        arrow::field("vectors", arrow::list(physical_vector)),
        arrow::field("by_name", arrow::map(arrow::utf8(), physical_vector)),
    }));
    const std::string json = R"([[[[1.0, 2.0], null], [["a", [3.0, 4.0]]]],
                                 [null, [["b", null]]]])";
    auto physical_array =
        arrow::ipc::internal::json::ArrayFromJSON(physical_type, json).ValueOrDie();
    RoaringBitmap32 bitmap;
    bitmap.Add(1);
    auto mock_reader = std::make_unique<MockFileBatchReader>(physical_array, physical_type, bitmap,
                                                             /*read_batch_size=*/10);
    mock_reader->EnableRandomizeBatchSize(false);
    VectorFileBatchReader reader(std::move(mock_reader), GetDefaultPool());
    ArrowSchema c_read_schema;
    ASSERT_TRUE(arrow::ExportSchema(*arrow::schema(logical_type->fields()), &c_read_schema).ok());
    ASSERT_OK(reader.SetReadSchema(&c_read_schema, /*predicate=*/nullptr,
                                   /*selection_bitmap=*/std::nullopt));

    ASSERT_OK_AND_ASSIGN(BatchReader::ReadBatchWithBitmap batch_with_bitmap,
                         reader.NextBatchWithBitmap());
    ASSERT_FALSE(batch_with_bitmap.second.Contains(0));
    ASSERT_TRUE(batch_with_bitmap.second.Contains(1));
    arrow::Result<std::shared_ptr<arrow::Array>> actual_result = arrow::ImportArray(
        batch_with_bitmap.first.first.get(), batch_with_bitmap.first.second.get());
    ASSERT_TRUE(actual_result.ok()) << actual_result.status().ToString();
    std::shared_ptr<arrow::Array> actual = std::move(actual_result).ValueOrDie();
    auto expected = arrow::ipc::internal::json::ArrayFromJSON(logical_type, json).ValueOrDie();
    ASSERT_TRUE(expected->Equals(actual)) << actual->ToString();
}

TEST(VectorFileBatchReaderTest, RejectInvalidVectorValues) {
    auto physical_type =
        AsStructType(arrow::struct_({arrow::field("embedding", arrow::list(arrow::float32()))}));
    auto logical_type = AsStructType(
        arrow::struct_({arrow::field("embedding", arrow::fixed_size_list(arrow::float32(), 3))}));
    for (const char* json : {R"([[[1.0, 2.0]]])", R"([[[1.0, null, 3.0]]])"}) {
        auto physical_array =
            arrow::ipc::internal::json::ArrayFromJSON(physical_type, json).ValueOrDie();
        auto mock_reader = std::make_unique<MockFileBatchReader>(physical_array, physical_type,
                                                                 /*read_batch_size=*/10);
        mock_reader->EnableRandomizeBatchSize(false);
        VectorFileBatchReader reader(std::move(mock_reader), GetDefaultPool());
        ArrowSchema c_read_schema;
        ASSERT_TRUE(
            arrow::ExportSchema(*arrow::schema(logical_type->fields()), &c_read_schema).ok());
        ASSERT_OK(reader.SetReadSchema(&c_read_schema, /*predicate=*/nullptr,
                                       /*selection_bitmap=*/std::nullopt));
        ASSERT_NOK(reader.NextBatch());
    }
}

TEST(VectorFileBatchReaderTest, RejectInvalidFixedSizeListVectorValues) {
    auto values_builder = std::make_shared<arrow::FloatBuilder>();
    arrow::FixedSizeListBuilder vector_builder(arrow::default_memory_pool(), values_builder, 3);
    ASSERT_TRUE(values_builder->Append(1.0f).ok());
    ASSERT_TRUE(values_builder->AppendNull().ok());
    ASSERT_TRUE(values_builder->Append(3.0f).ok());
    ASSERT_TRUE(vector_builder.Append().ok());
    std::shared_ptr<arrow::Array> vector_array;
    ASSERT_TRUE(vector_builder.Finish(&vector_array).ok());
    arrow::Result<std::shared_ptr<arrow::StructArray>> struct_result =
        arrow::StructArray::Make({vector_array}, {"embedding"});
    ASSERT_TRUE(struct_result.ok()) << struct_result.status().ToString();
    std::shared_ptr<arrow::StructArray> physical_array = std::move(struct_result).ValueOrDie();
    auto physical_type = AsStructType(physical_array->type());
    auto mock_reader = std::make_unique<MockFileBatchReader>(physical_array, physical_type,
                                                             /*read_batch_size=*/10);
    mock_reader->EnableRandomizeBatchSize(false);
    VectorFileBatchReader reader(std::move(mock_reader), GetDefaultPool());
    ArrowSchema c_read_schema;
    ASSERT_TRUE(arrow::ExportSchema(*arrow::schema(physical_type->fields()), &c_read_schema).ok());
    ASSERT_OK(reader.SetReadSchema(&c_read_schema, /*predicate=*/nullptr,
                                   /*selection_bitmap=*/std::nullopt));
    ASSERT_NOK(reader.NextBatch());
}

}  // namespace paimon::test
