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
#include <unordered_map>
#include <vector>

#include "orc/Type.hh"
#include "orc/sargs/Literal.hh"
#include "orc/sargs/SearchArgument.hh"
#include "paimon/predicate/compound_predicate.h"
#include "paimon/predicate/leaf_predicate.h"
#include "paimon/predicate/predicate.h"
#include "paimon/result.h"
#include "paimon/status.h"

namespace orc {
class Type;
}  // namespace orc

namespace paimon {
class CompoundPredicate;
class Function;
class LeafPredicate;
class Literal;
class Predicate;
enum class FieldType;
}  // namespace paimon

namespace paimon::orc {

class PredicateConverter {
 public:
    PredicateConverter() = delete;
    ~PredicateConverter() = delete;

    // convert paimon predicate to orc SearchArgument
    static Result<std::unique_ptr<::orc::SearchArgument>> Convert(
        const ::orc::Type& orc_type, const std::shared_ptr<Predicate>& predicate);

 private:
    static Status ConvertCompound(const std::shared_ptr<CompoundPredicate>& compound_predicate,
                                  const std::unordered_map<std::string, uint64_t>& name_id_mapping,
                                  ::orc::SearchArgumentBuilder* builder);

    static Status ConvertLeaf(const std::shared_ptr<LeafPredicate>& leaf_predicate,
                              const std::unordered_map<std::string, uint64_t>& name_id_mapping,
                              ::orc::SearchArgumentBuilder* builder);

    static Status Convert(const std::shared_ptr<Predicate>& predicate,
                          const std::unordered_map<std::string, uint64_t>& name_id_mapping,
                          ::orc::SearchArgumentBuilder* builder);

    static Result<bool> IsPredicatePushable(
        const std::shared_ptr<Predicate>& predicate,
        const std::unordered_map<std::string, uint64_t>& name_id_mapping);

    static Result<bool> IsCompoundPredicatePushable(
        const std::shared_ptr<CompoundPredicate>& compound_predicate,
        const std::unordered_map<std::string, uint64_t>& name_id_mapping);

    static Result<bool> IsLeafPredicatePushable(
        const std::shared_ptr<LeafPredicate>& leaf_predicate,
        const std::unordered_map<std::string, uint64_t>& name_id_mapping);

    static Status CheckLiteralNotEmpty(const std::vector<Literal>& literals,
                                       const Function& function, const std::string& field_name);

    static Result<::orc::PredicateDataType> ConvertToOrcPredicateType(const FieldType& field_type);

    static Result<std::vector<::orc::Literal>> ConvertToOrcLiterals(
        const std::vector<Literal>& literals);
};

}  // namespace paimon::orc
