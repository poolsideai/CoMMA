// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "c-helpers.h"

#include "stdarg.h"
#include "stdio.h"
#include "string.h"

#define MAX_BUF_SZ 256

inner_logger_t inner_logger;

void logger_helper(
    ncclDebugLogLevel level,
    unsigned long flags, const char *file, int line,
    const char *fmt, ...) {
  char buf[MAX_BUF_SZ];
  va_list args;
  va_start(args, fmt);
  int n = vsnprintf(buf, MAX_BUF_SZ, fmt, args);
  inner_logger(level, flags, file, line, buf, n);
  va_end(args);
}

void init_logger(inner_logger_t inner) {
  inner_logger = inner;
}
