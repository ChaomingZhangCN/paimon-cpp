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

#include "paimon/format/parquet/parquet_vector_converter.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include "arrow/array.h"
#include "arrow/array/array_nested.h"
#include "arrow/array/builder_primitive.h"
#include "arrow/compute/api.h"
#include "arrow/type.h"
#include "paimon/common/utils/arrow/status_utils.h"
#include "paimon/common/utils/checked_cast.h"
#include "paimon/status.h"

namespace paimon::parquet {
namespace {

bool ContainsVectorType(const std::shared_ptr<arrow::DataType>& type) {
    if (type->id() == arrow::Type::FIXED_SIZE_LIST) {
        return true;
    }
    for (const auto& field : type->fields()) {
        if (ContainsVectorType(field->type())) {
            return true;
        }
    }
    return false;
}

Status ValidateVectorElements(const arrow::FixedSizeListArray& array, int32_t vector_length) {
    const std::shared_ptr<arrow::Array>& values = array.values();
    if (values->null_count() == 0) {
        return Status::OK();
    }
    for (int64_t i = 0; i < array.length(); ++i) {
        if (array.IsNull(i)) {
            continue;
        }
        int64_t value_offset = (array.offset() + i) * vector_length;
        for (int32_t j = 0; j < vector_length; ++j) {
            if (values->IsNull(value_offset + j)) {
                return Status::Invalid("VECTOR cannot contain null elements");
            }
        }
    }
    return Status::OK();
}

Result<int64_t> GetIndexCapacity(int64_t row_count, int32_t vector_length) {
    if (vector_length < 1) {
        return Status::Invalid("VECTOR length must be positive");
    }
    if (row_count > std::numeric_limits<int64_t>::max() / vector_length) {
        return Status::Invalid("VECTOR values exceed the supported Arrow array length");
    }
    return row_count * vector_length;
}

Result<std::shared_ptr<arrow::Array>> ConvertVectorToList(
    const std::shared_ptr<arrow::Array>& array, arrow::MemoryPool* pool) {
    const auto& vector_array = checked_cast<const arrow::FixedSizeListArray&>(*array);
    const auto& vector_type = checked_cast<const arrow::FixedSizeListType&>(*array->type());
    int32_t vector_length = vector_type.list_size();
    PAIMON_RETURN_NOT_OK(ValidateVectorElements(vector_array, vector_length));

    arrow::Int32Builder offsets_builder(pool);
    arrow::Int64Builder indices_builder(pool);
    arrow::BooleanBuilder validity_builder(pool);
    PAIMON_RETURN_NOT_OK_FROM_ARROW(offsets_builder.Reserve(vector_array.length() + 1));
    PAIMON_ASSIGN_OR_RAISE(int64_t index_capacity,
                           GetIndexCapacity(vector_array.length(), vector_length));
    PAIMON_RETURN_NOT_OK_FROM_ARROW(indices_builder.Reserve(index_capacity));
    PAIMON_RETURN_NOT_OK_FROM_ARROW(validity_builder.Reserve(vector_array.length()));
    PAIMON_RETURN_NOT_OK_FROM_ARROW(offsets_builder.Append(0));

    int32_t offset = 0;
    for (int64_t i = 0; i < vector_array.length(); ++i) {
        bool valid = !vector_array.IsNull(i);
        PAIMON_RETURN_NOT_OK_FROM_ARROW(validity_builder.Append(valid));
        if (valid) {
            if (vector_length > std::numeric_limits<int32_t>::max() - offset) {
                return Status::Invalid("VECTOR values exceed the maximum Parquet LIST offset");
            }
            int64_t value_offset = (vector_array.offset() + i) * vector_length;
            for (int32_t j = 0; j < vector_length; ++j) {
                PAIMON_RETURN_NOT_OK_FROM_ARROW(indices_builder.Append(value_offset + j));
            }
            offset += vector_length;
        }
        PAIMON_RETURN_NOT_OK_FROM_ARROW(offsets_builder.Append(offset));
    }

    std::shared_ptr<arrow::Array> offsets;
    std::shared_ptr<arrow::Array> indices;
    std::shared_ptr<arrow::Array> validity;
    PAIMON_RETURN_NOT_OK_FROM_ARROW(offsets_builder.Finish(&offsets));
    PAIMON_RETURN_NOT_OK_FROM_ARROW(indices_builder.Finish(&indices));
    PAIMON_RETURN_NOT_OK_FROM_ARROW(validity_builder.Finish(&validity));

    arrow::compute::ExecContext exec_context(pool);
    PAIMON_ASSIGN_OR_RAISE_FROM_ARROW(
        arrow::Datum values,
        arrow::compute::Take(arrow::Datum(vector_array.values()), arrow::Datum(indices),
                             arrow::compute::TakeOptions::NoBoundsCheck(), &exec_context));
    std::shared_ptr<arrow::Buffer> null_bitmap;
    if (vector_array.null_count() != 0) {
        null_bitmap = validity->data()->buffers[1];
    }
    std::shared_ptr<arrow::DataType> write_type =
        arrow::list(vector_type.value_field()->WithType(values.type()));
    return std::make_shared<arrow::ListArray>(write_type, vector_array.length(),
                                              offsets->data()->buffers[1], values.make_array(),
                                              null_bitmap, vector_array.null_count());
}

}  // namespace

std::shared_ptr<arrow::DataType> ParquetVectorConverter::GetWriteType(
    const std::shared_ptr<arrow::DataType>& logical_type) {
    switch (logical_type->id()) {
        case arrow::Type::FIXED_SIZE_LIST: {
            const auto& vector_type = checked_cast<const arrow::FixedSizeListType&>(*logical_type);
            return arrow::list(
                vector_type.value_field()->WithType(GetWriteType(vector_type.value_type())));
        }
        case arrow::Type::STRUCT: {
            arrow::FieldVector fields;
            fields.reserve(logical_type->num_fields());
            for (const auto& field : logical_type->fields()) {
                fields.push_back(field->WithType(GetWriteType(field->type())));
            }
            return arrow::struct_(fields);
        }
        case arrow::Type::LIST:
            return arrow::list(
                logical_type->field(0)->WithType(GetWriteType(logical_type->field(0)->type())));
        case arrow::Type::MAP: {
            const auto& map_type = checked_cast<const arrow::MapType&>(*logical_type);
            return std::make_shared<arrow::MapType>(
                map_type.key_field()->WithType(GetWriteType(map_type.key_type())),
                map_type.item_field()->WithType(GetWriteType(map_type.item_type())),
                map_type.keys_sorted());
        }
        default:
            return logical_type;
    }
}

Result<std::shared_ptr<arrow::Array>> ParquetVectorConverter::ConvertToWriteType(
    const std::shared_ptr<arrow::Array>& array, arrow::MemoryPool* pool) {
    if (!ContainsVectorType(array->type())) {
        return array;
    }
    if (array->type()->id() == arrow::Type::FIXED_SIZE_LIST) {
        return ConvertVectorToList(array, pool);
    }
    switch (array->type()->id()) {
        case arrow::Type::STRUCT:
        case arrow::Type::LIST:
        case arrow::Type::MAP: {
            std::vector<std::shared_ptr<arrow::ArrayData>> children;
            children.reserve(array->type()->num_fields());
            for (const auto& child_data : array->data()->child_data) {
                PAIMON_ASSIGN_OR_RAISE(std::shared_ptr<arrow::Array> child,
                                       ConvertToWriteType(arrow::MakeArray(child_data), pool));
                children.push_back(child->data());
            }
            std::shared_ptr<arrow::ArrayData> data = array->data()->Copy();
            data->child_data = std::move(children);
            data->type = GetWriteType(array->type());
            return arrow::MakeArray(data);
        }
        default:
            return array;
    }
}

}  // namespace paimon::parquet
