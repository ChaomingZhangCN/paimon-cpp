/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "arrow/api.h"
#include "orc/Common.hh"
#include "orc/Statistics.hh"
#include "orc/Type.hh"
#include "paimon/format/column_stats.h"
#include "paimon/format/format_stats_extractor.h"
#include "paimon/result.h"
#include "paimon/type_fwd.h"

namespace arrow {
class DataType;
class Schema;
}  // namespace arrow

namespace orc {
class ColumnStatistics;
class Type;
}  // namespace orc

namespace paimon {
class FileSystem;
class MemoryPool;
}  // namespace paimon

namespace paimon::orc {
class OrcStatsExtractor : public FormatStatsExtractor {
 public:
    explicit OrcStatsExtractor(const std::shared_ptr<arrow::Schema>& write_schema)
        : write_schema_(write_schema) {}
    Result<ColumnStatsVector> Extract(const std::shared_ptr<FileSystem>& file_system,
                                      const std::string& path,
                                      const std::shared_ptr<MemoryPool>& pool) override {
        PAIMON_ASSIGN_OR_RAISE(auto result, ExtractWithFileInfo(file_system, path, pool));
        return result.first;
    }

    Result<std::pair<ColumnStatsVector, FileInfo>> ExtractWithFileInfo(
        const std::shared_ptr<FileSystem>& file_system, const std::string& path,
        const std::shared_ptr<MemoryPool>& pool) override;

 private:
    Result<std::unique_ptr<ColumnStats>> FetchColumnStatistics(
        const ::orc::ColumnStatistics* column_stats, const ::orc::Type* type,
        const std::shared_ptr<arrow::DataType>& write_type) const;

 private:
    int64_t row_count_ = -1;
    std::shared_ptr<arrow::Schema> write_schema_;
};

}  // namespace paimon::orc
