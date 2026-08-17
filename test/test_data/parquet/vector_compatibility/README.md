<!--
  ~ Licensed to the Apache Software Foundation (ASF) under one
  ~ or more contributor license agreements.  See the NOTICE file
  ~ distributed with this work for additional information
  ~ regarding copyright ownership.  The ASF licenses this file
  ~ to you under the Apache License, Version 2.0 (the
  ~ "License"); you may not use this file except in compliance
  ~ with the License.  You may obtain a copy of the License at
  ~
  ~   http://www.apache.org/licenses/LICENSE-2.0
  ~
  ~ Unless required by applicable law or agreed to in writing,
  ~ software distributed under the License is distributed on an
  ~ "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  ~ KIND, either express or implied.  See the License for the
  ~ specific language governing permissions and limitations
  ~ under the License.
-->

# VECTOR Parquet compatibility fixtures

These files pin the two physical Arrow schemas produced by Java and Rust writers for Paimon
VECTOR columns.

- `java_vector.parquet` was copied from Apache Paimon Rust commit
  `403a2b2e9bfc4ea66cd7e633619f1460efd18bc8`, path
  `crates/paimon/testdata/pkvector/pk_vector_ivf_flat/bucket-0/data-932a1249-f7e0-4a03-8e1f-ab8c85cbb76f-0.parquet`.
  The fixture documentation records Apache Paimon Java commit `7234e4c34` and
  `PkVectorFixtureGenerator` as its source. Its VECTOR column is exposed as Arrow `list`.
- `rust_vector.parquet` was generated with Apache Arrow Rust 58.4.0 using
  `FixedSizeListBuilder<Float32Builder>` and `parquet::arrow::ArrowWriter`, the same Arrow and
  Parquet representation used by Apache Paimon Rust. Its VECTOR column is exposed as Arrow
  `fixed_size_list[3]`. The rows are `(1, [1, 2, 3])`, `(2, [7, 8, 9])`, and
  `(3, [4, 5, 6])`.

SHA-256 checksums:

```text
2b2325cc2266301beaa2c78ec666cb5e0ee62283049de2a7231e3c9ae07bf3ca  java_vector.parquet
b5ba47e766ad72fca9c8485aa718ad27709c1fb4d34fb3670aa35e2001cbdbb0  rust_vector.parquet
```
