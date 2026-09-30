"""Read the pending batch's queue without starting a Modal function."""
import argparse
import asyncio
import json

from modal.client import _Client
from modal_proto import api_pb2


async def read_status(app_id):
    client = await _Client.from_env()
    layout = await client.stub.AppGetLayout(api_pb2.AppGetLayoutRequest(app_id=app_id))
    function_id = layout.app_layout.function_ids["run_cli"]
    stats = await client.stub.FunctionGetCurrentStats(api_pb2.FunctionGetCurrentStatsRequest(function_id=function_id))
    return {"app_id": app_id, "function_id": function_id, "backlog": stats.backlog,
            "total_tasks": stats.num_total_tasks, "running_inputs": stats.num_running_inputs}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--app-id", required=True)
    args = p.parse_args()
    print(json.dumps(asyncio.run(read_status(args.app_id))))
