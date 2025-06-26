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

#ifndef __PROFILER_C_HELPER_H__
#define __PROFILER_C_HELPER_H__

typedef enum {
    NCCL_LOG_NONE = 0,
    NCCL_LOG_VERSION = 1,
    NCCL_LOG_WARN = 2,
    NCCL_LOG_INFO = 3,
    NCCL_LOG_ABORT = 4,
    NCCL_LOG_TRACE = 5
} ncclDebugLogLevel;

void logger_helper(
    ncclDebugLogLevel level,
    unsigned long flags, const char *file, int line,
    const char *fmt, ...);

typedef void (*inner_logger_t)(ncclDebugLogLevel /* level */,
                               unsigned long     /* flags */,
                               const char*       /* file */,
                               int               /* line */,
                               const char*       /* message */,
                               int               /* length */);

void init_logger(inner_logger_t inner);


#endif
