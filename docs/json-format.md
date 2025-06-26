# CoMMA JSON export format
CoMMA exports telemetry as JSON objects when `NCCL_PROFILER_LATENCY_FILE` is set.

## JSON object fields
The JSON output file is list of objects for each event, with the following key-value pairs:

- args (Object): contains arguments related to the communication operation.
   - size (Integer): the total size of the data involved in the communication, in bytes.
- cat (String): a category identifier for the operation (e.g., "COLL" for collective communication, "PROXY" for proxy operations).
- comm_hash (String): a hexadecimal string representing a hash value that uniquely identifies the NCCL communicator.
- dur (Integer): the duration of the operation, in microseconds. For collective events, this often represents the enqueue duration .
- name (String): the name of the communication operation (e.g., "all_reduce", "ProxyOp").
- ph (String): a phase identifier, typically "X" indicating a complete event with a start time and duration.
- ts (Integer): the timestamp of the operation's start, in microseconds.
- rank (Integer): the rank of the process performing the communication operation.
- pid (Integer): the process ID of the originating process.

## Collective specific fields
- child_dur (Integer, Optional): the execution duration of child operations (like proxy operations) associated with this collective, in microseconds. This field helps determine the actual execution time of the collective, as the dur field often reflects enqueue time.
- seq_num (Integer): a sequence number for this specific collective operation within its communicator.
- proxyops (Array of Objects, Optional): an array of objects describing proxy operations associated with this collective communication. Each object in this array follows the “ProxyOp specific fields” structure.

## ProxyOp specific fields
Proxy operation events ("cat":"PROXY") have the following additional fields:

- chunk_size (Integer): the size of the data chunk processed by this proxy operation, in bytes t.
- is_send (Boolean): indicates whether the proxy operation is a send (true) or receive (false).
- n_steps (Integer): the number of steps (network transfers) within this proxy operation.
- peer (Integer): the peer rank involved in this proxy operation.
- steps (Array of Objects): an array describing individual network transfer steps within the proxy operation. Each step object contains:
- start_time (Integer): The start time of the step, in microseconds.
- end_time (Integer): The end time of the step, in microseconds.
- size (Integer): The size of data processed in this step, in bytes.
- step (Integer): The step number within the proxy operation.

## Sample output
The following sample JSON output snippet is captured by CoMMA and represents a trace event for an `all_reduce` collective operation.

```(json)
{
  "args":{"size":20971520},
  "cat":"COLL",
  "child_dur":1456,
  "comm_hash":"0x58aecebabb9e37af",
  "dur":11039,
  "name":"all_reduce",
  "ph":"X",
  "rank":0,
  "seq_num":57,
  "ts":9615369,
  "proxyops":[
    {
      "cat":"PROXY","chunk_size":1048576,"dur":1379,"is_send":false,
      "n_steps":2,"name":"ProxyOp","peer":16,"ph":"X","pid":169,
      "ts":9624951,
      "steps":[
        {"end_time":9626229,"size":1048576,"start_time":9624952,"step":1},
        {"end_time":9626229,"size":262144,"start_time":9624953,"step":2}
      ]
    },
    {
      "cat":"PROXY","chunk_size":1048576,"dur":1382,"is_send":true,
      "n_steps":2,"name":"ProxyOp","peer":16,"ph":"X","pid":169,
      "ts":9625026,
      "steps":[
        {"end_time":9626408,"size":1048576,"start_time":9626308,"step":1},
        {"end_time":9626408,"size":262144,"start_time":9626332,"step":2}
      ]
    }
  ]
}
```
In the preceding sample output , the `all_reduce` operation aggregates data from multiple participants and distributes the result to all of them.

The `proxyops` array contains details
about the individual send and receive operations that comprise the `all_reduce` operation. Each `ProxyOp` object describes a data transfer, including its duration, chunk size, and whether it's a send or receive operation.

The `steps` array within each `ProxyOp` further breaks down the transfer into smaller, sequential steps, providing start and end times, and the size of data transferred in each step. 


This detailed tracing helps Google Cloud to collect data that is useful for understanding the performance and behavior of distributed collective operations.
