"""MCP Server for Few-Shot MMR Diversity Selector.
100% Python Standard Library JSON-RPC.
"""
import sys
import json
from client import FewShotMmrDiversitySelector

def main():
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        try:
            req = json.loads(line)
            method = req.get("method")
            msg_id = req.get("id")

            if method == "tools/list":
                resp = {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "tools": [
                            {
                                "name": "select_exemplars",
                                "description": "Select diverse, high-relevance few-shot examples using Maximal Marginal Relevance",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "query": {"type": "string", "description": "Target prompt query"},
                                        "pool": {"type": "array", "description": "List of exemplar objects with input and output"},
                                        "k": {"type": "integer", "default": 3},
                                        "lambda_param": {"type": "number", "default": 0.5}
                                    },
                                    "required": ["query", "pool"]
                                }
                            }
                        ]
                    }
                }
            elif method == "tools/call":
                args = req.get("params", {}).get("arguments", {})
                query = args.get("query", "")
                pool = args.get("pool", [])
                k = args.get("k", 3)
                lp = args.get("lambda_param", 0.5)
                res = FewShotMmrDiversitySelector.select_exemplars(query, pool, k=k, lambda_param=lp)
                resp = {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {"content": [{"type": "text", "text": json.dumps(res, indent=2)}]}
                }
            else:
                resp = {"jsonrpc": "2.0", "id": msg_id, "error": {"code": -32601, "message": "Method not found"}}
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()
        except Exception as e:
            sys.stdout.write(json.dumps({"jsonrpc": "2.0", "error": {"code": -32000, "message": str(e)}}) + "\n")
            sys.stdout.flush()

if __name__ == "__main__":
    main()
