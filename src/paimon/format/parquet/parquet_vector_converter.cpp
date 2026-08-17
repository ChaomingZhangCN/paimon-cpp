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
#include <memory>

#include "arrow/array.h"
#include "arrow/array/array_nested.h"
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

Status ValidateVectorElements(const std::shared_ptr<arrow::Array>& array) {
    if (!ContainsVectorType(array->type())) {
        return Status::OK();
    }
    if (array->type_id() == arrow::Type::FIXED_SIZE_LIST) {
        const auto& vector_array = checked_cast<const arrow::FixedSizeListArray&>(*array);
        const auto& vector_type = checked_cast<const arrow::FixedSizeListType&>(*array->type());
        const std::shared_ptr<arrow::Array>& values = vector_array.values();
        if (values->null_count() == 0) {
            return Status::OK();
        }
        for (int64_t i = 0; i < vector_array.length(); ++i) {
            if (vector_array.IsNull(i)) {
                continue;
            }
            int64_t value_offset = (vector_array.offset() + i) * vector_type.list_size();
            for (int32_t j = 0; j < vector_type.list_size(); ++j) {
                if (values->IsNull(value_offset + j)) {
                    return Status::Invalid("VECTOR cannot contain null elements");
                }
            }
        }
        return Status::OK();
    }
    for (const auto& child_data : array->data()->child_data) {
        PAIMON_RETURN_NOT_OK(ValidateVectorElements(arrow::MakeArray(child_data)));
    }
    return Status::OK();
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
    PAIMON_RETURN_NOT_OK(ValidateVectorElements(array));
    std::shared_ptr<arrow::DataType> write_type = GetWriteType(array->type());
    arrow::compute::ExecContext exec_context(pool);
    arrow::TypeHolder type_holder(write_type.get());
    arrow::compute::CastOptions options = arrow::compute::CastOptions::Safe();
    PAIMON_ASSIGN_OR_RAISE_FROM_ARROW(
        std::shared_ptr<arrow::Array> result,
        arrow::compute::Cast(*array, type_holder, options, &exec_context));
    return result;
}

}  // namespace paimon::parquet
