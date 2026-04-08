#!/bin/bash
# ==========================================================================
# Full Agent Protocol API test suite using curl
#
# Prerequisites:
#   python test_server.py   # start test server on port 7778
#
# Usage:
#   bash test_curl.sh
#   AP_BASE_URL=http://other-host:8000/ap bash test_curl.sh
# ==========================================================================

set -euo pipefail

BASE_URL="${AP_BASE_URL:-http://localhost:7778/ap}"
PASS=0
FAIL=0
TOTAL=0

# Colors
green() { echo -e "\033[32m$1\033[0m"; }
red()   { echo -e "\033[31m$1\033[0m"; }
bold()  { echo -e "\033[1m$1\033[0m"; }

assert_status() {
    local name="$1" expected="$2" actual="$3"
    TOTAL=$((TOTAL + 1))
    if [ "$actual" = "$expected" ]; then
        green "  PASS: $name (HTTP $actual)"
        PASS=$((PASS + 1))
    else
        red "  FAIL: $name (expected $expected, got $actual)"
        FAIL=$((FAIL + 1))
    fi
}

assert_json() {
    local name="$1" json="$2" jq_expr="$3" expected="$4"
    TOTAL=$((TOTAL + 1))
    actual=$(echo "$json" | jq -r "$jq_expr" 2>/dev/null || echo "JQ_ERROR")
    if [ "$actual" = "$expected" ]; then
        green "  PASS: $name = $actual"
        PASS=$((PASS + 1))
    else
        red "  FAIL: $name (expected '$expected', got '$actual')"
        FAIL=$((FAIL + 1))
    fi
}

assert_not_empty() {
    local name="$1" json="$2" jq_expr="$3"
    TOTAL=$((TOTAL + 1))
    actual=$(echo "$json" | jq -r "$jq_expr" 2>/dev/null || echo "")
    if [ -n "$actual" ] && [ "$actual" != "null" ] && [ "$actual" != "" ]; then
        green "  PASS: $name is set ($actual)"
        PASS=$((PASS + 1))
    else
        red "  FAIL: $name is empty or null"
        FAIL=$((FAIL + 1))
    fi
}

# ==========================================================================
bold "=========================================="
bold "Agent Protocol API Test Suite"
bold "Base URL: $BASE_URL"
bold "=========================================="
echo

# ==========================================================================
bold "1. AGENT INTROSPECTION"
# ==========================================================================

# POST /agents/search
RESP=$(curl -s -w "\n%{http_code}" -X POST "$BASE_URL/agents/search" -H 'Content-Type: application/json' -d '{}')
BODY=$(echo "$RESP" | head -n -1)
STATUS=$(echo "$RESP" | tail -1)
assert_status "POST /agents/search" 200 "$STATUS"
assert_json "agents list length >= 1" "$BODY" '. | length >= 1' "true"
assert_json "first agent has agent_id" "$BODY" '.[0].agent_id' "echo_agent"

# GET /agents/{agent_id}
RESP=$(curl -s -w "\n%{http_code}" "$BASE_URL/agents/echo_agent")
BODY=$(echo "$RESP" | head -n -1)
STATUS=$(echo "$RESP" | tail -1)
assert_status "GET /agents/echo_agent" 200 "$STATUS"
assert_json "agent name" "$BODY" '.name' "echo-agent"
assert_json "agent capabilities messages" "$BODY" '.capabilities["ap.io.messages"]' "true"

# GET /agents/{agent_id} -- not found
STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/agents/nonexistent")
assert_status "GET /agents/nonexistent (404)" 404 "$STATUS"

# GET /agents/{agent_id}/schemas
RESP=$(curl -s -w "\n%{http_code}" "$BASE_URL/agents/echo_agent/schemas")
BODY=$(echo "$RESP" | head -n -1)
STATUS=$(echo "$RESP" | tail -1)
assert_status "GET /agents/echo_agent/schemas" 200 "$STATUS"
assert_json "has input_schema" "$BODY" '.input_schema.type' "object"

echo

# ==========================================================================
bold "2. THREAD CRUD"
# ==========================================================================

# POST /threads -- create
RESP=$(curl -s -w "\n%{http_code}" -X POST "$BASE_URL/threads" -H 'Content-Type: application/json' -d '{"metadata":{"test":"curl"}}')
BODY=$(echo "$RESP" | head -n -1)
STATUS=$(echo "$RESP" | tail -1)
assert_status "POST /threads (create)" 200 "$STATUS"
THREAD_ID=$(echo "$BODY" | jq -r '.thread_id')
assert_not_empty "thread_id" "$BODY" '.thread_id'
assert_json "status is idle" "$BODY" '.status' "idle"

# GET /threads/{thread_id}
RESP=$(curl -s -w "\n%{http_code}" "$BASE_URL/threads/$THREAD_ID")
BODY=$(echo "$RESP" | head -n -1)
STATUS=$(echo "$RESP" | tail -1)
assert_status "GET /threads/{id}" 200 "$STATUS"
assert_json "thread_id matches" "$BODY" '.thread_id' "$THREAD_ID"

# PATCH /threads/{thread_id}
RESP=$(curl -s -w "\n%{http_code}" -X PATCH "$BASE_URL/threads/$THREAD_ID" -H 'Content-Type: application/json' -d '{"metadata":{"updated":true}}')
BODY=$(echo "$RESP" | head -n -1)
STATUS=$(echo "$RESP" | tail -1)
assert_status "PATCH /threads/{id}" 200 "$STATUS"
assert_json "updated metadata" "$BODY" '.metadata.updated' "true"

# POST /threads/{thread_id}/copy
RESP=$(curl -s -w "\n%{http_code}" -X POST "$BASE_URL/threads/$THREAD_ID/copy" -H 'Content-Type: application/json')
BODY=$(echo "$RESP" | head -n -1)
STATUS=$(echo "$RESP" | tail -1)
assert_status "POST /threads/{id}/copy" 200 "$STATUS"
COPIED_ID=$(echo "$BODY" | jq -r '.thread_id')
assert_json "copy has different id" "$BODY" ".thread_id != \"$THREAD_ID\"" "true"

# POST /threads/search
RESP=$(curl -s -w "\n%{http_code}" -X POST "$BASE_URL/threads/search" -H 'Content-Type: application/json' -d '{}')
BODY=$(echo "$RESP" | head -n -1)
STATUS=$(echo "$RESP" | tail -1)
assert_status "POST /threads/search" 200 "$STATUS"
assert_json "search returns >= 2" "$BODY" '. | length >= 2' "true"

# GET /threads/{thread_id}/history
RESP=$(curl -s -w "\n%{http_code}" "$BASE_URL/threads/$THREAD_ID/history")
BODY=$(echo "$RESP" | head -n -1)
STATUS=$(echo "$RESP" | tail -1)
assert_status "GET /threads/{id}/history" 200 "$STATUS"
assert_json "history is array" "$BODY" '. | type' "array"
assert_json "history has checkpoint" "$BODY" '.[0].checkpoint.thread_id' "$THREAD_ID"

# DELETE /threads/{thread_id} -- delete copied thread
STATUS=$(curl -s -o /dev/null -w "%{http_code}" -X DELETE "$BASE_URL/threads/$COPIED_ID")
assert_status "DELETE /threads/{copied_id}" 200 "$STATUS"

# GET deleted thread -- 404
STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/threads/$COPIED_ID")
assert_status "GET deleted thread (404)" 404 "$STATUS"

echo

# ==========================================================================
bold "3. THREAD-SCOPED RUNS"
# ==========================================================================

# POST /threads/{thread_id}/runs -- create run
RESP=$(curl -s -w "\n%{http_code}" -X POST "$BASE_URL/threads/$THREAD_ID/runs" -H 'Content-Type: application/json' \
    -d "{\"assistant_id\":\"echo_agent\",\"input\":{\"messages\":[{\"role\":\"user\",\"content\":\"Hello from curl\"}]}}")
BODY=$(echo "$RESP" | head -n -1)
STATUS=$(echo "$RESP" | tail -1)
assert_status "POST /threads/{id}/runs (create)" 200 "$STATUS"
RUN_ID=$(echo "$BODY" | jq -r '.run_id')
assert_not_empty "run_id" "$BODY" '.run_id'
assert_json "run thread_id" "$BODY" '.thread_id' "$THREAD_ID"

# Wait for run to complete
sleep 1

# GET /threads/{thread_id}/runs/{run_id} -- get run
RESP=$(curl -s -w "\n%{http_code}" "$BASE_URL/threads/$THREAD_ID/runs/$RUN_ID")
BODY=$(echo "$RESP" | head -n -1)
STATUS=$(echo "$RESP" | tail -1)
assert_status "GET /threads/{id}/runs/{rid}" 200 "$STATUS"
assert_json "run status" "$BODY" '.status' "success"

# GET /threads/{thread_id}/runs -- list runs
RESP=$(curl -s -w "\n%{http_code}" "$BASE_URL/threads/$THREAD_ID/runs")
BODY=$(echo "$RESP" | head -n -1)
STATUS=$(echo "$RESP" | tail -1)
assert_status "GET /threads/{id}/runs (list)" 200 "$STATUS"
assert_json "runs list >= 1" "$BODY" '. | length >= 1' "true"

# GET /threads/{thread_id}/runs/{run_id}/join
RESP=$(curl -s -w "\n%{http_code}" "$BASE_URL/threads/$THREAD_ID/runs/$RUN_ID/join")
BODY=$(echo "$RESP" | head -n -1)
STATUS=$(echo "$RESP" | tail -1)
assert_status "GET /threads/{id}/runs/{rid}/join" 200 "$STATUS"
assert_not_empty "join has run" "$BODY" '.run.run_id'
assert_not_empty "join has values" "$BODY" '.values'

# POST /threads/{thread_id}/runs/{run_id}/cancel
# Create a new run first to cancel
RESP=$(curl -s -w "\n%{http_code}" -X POST "$BASE_URL/threads/$THREAD_ID/runs" -H 'Content-Type: application/json' \
    -d "{\"assistant_id\":\"echo_agent\",\"input\":{\"messages\":[{\"role\":\"user\",\"content\":\"cancel me\"}]}}")
CANCEL_RID=$(echo "$RESP" | head -n -1 | jq -r '.run_id')
RESP=$(curl -s -w "\n%{http_code}" -X POST "$BASE_URL/threads/$THREAD_ID/runs/$CANCEL_RID/cancel" -H 'Content-Type: application/json')
BODY=$(echo "$RESP" | head -n -1)
STATUS=$(echo "$RESP" | tail -1)
assert_status "POST /threads/{id}/runs/{rid}/cancel" 200 "$STATUS"
assert_json "cancelled status" "$BODY" '.status' "interrupted"

# GET thread to verify values populated
RESP=$(curl -s -w "\n%{http_code}" "$BASE_URL/threads/$THREAD_ID")
BODY=$(echo "$RESP" | head -n -1)
assert_json "thread has messages" "$BODY" '.values.messages | length >= 1' "true"

echo

# ==========================================================================
bold "4. THREAD-SCOPED STREAMING & WAIT"
# ==========================================================================

# POST /threads/{thread_id}/runs/wait
RESP=$(curl -s -w "\n%{http_code}" -X POST "$BASE_URL/threads/$THREAD_ID/runs/wait" -H 'Content-Type: application/json' \
    -d "{\"assistant_id\":\"echo_agent\",\"input\":{\"messages\":[{\"role\":\"user\",\"content\":\"wait test\"}]}}")
BODY=$(echo "$RESP" | head -n -1)
STATUS=$(echo "$RESP" | tail -1)
assert_status "POST /threads/{id}/runs/wait" 200 "$STATUS"
assert_json "wait run status" "$BODY" '.run.status' "success"
assert_not_empty "wait has values" "$BODY" '.values'

# POST /threads/{thread_id}/runs/stream
RESP=$(curl -s -w "\n%{http_code}" -X POST "$BASE_URL/threads/$THREAD_ID/runs/stream" -H 'Content-Type: application/json' \
    -d "{\"assistant_id\":\"echo_agent\",\"input\":{\"messages\":[{\"role\":\"user\",\"content\":\"stream test\"}]}}")
BODY=$(echo "$RESP" | head -n -1)
STATUS=$(echo "$RESP" | tail -1)
assert_status "POST /threads/{id}/runs/stream" 200 "$STATUS"
# Check SSE events in body
TOTAL=$((TOTAL + 1))
if echo "$BODY" | grep -q "event: metadata"; then
    green "  PASS: stream has metadata event"
    PASS=$((PASS + 1))
else
    red "  FAIL: stream missing metadata event"
    FAIL=$((FAIL + 1))
fi
TOTAL=$((TOTAL + 1))
if echo "$BODY" | grep -q "event: end"; then
    green "  PASS: stream has end event"
    PASS=$((PASS + 1))
else
    red "  FAIL: stream missing end event"
    FAIL=$((FAIL + 1))
fi

echo

# ==========================================================================
bold "5. STATELESS RUNS"
# ==========================================================================

# POST /runs/wait
RESP=$(curl -s -w "\n%{http_code}" -X POST "$BASE_URL/runs/wait" -H 'Content-Type: application/json' \
    -d "{\"agent_id\":\"echo_agent\",\"input\":{\"messages\":[{\"role\":\"user\",\"content\":\"stateless wait\"}]}}")
BODY=$(echo "$RESP" | head -n -1)
STATUS=$(echo "$RESP" | tail -1)
assert_status "POST /runs/wait" 200 "$STATUS"
assert_json "runs/wait status" "$BODY" '.run.status' "success"
assert_not_empty "runs/wait has values" "$BODY" '.values'

# POST /runs/stream
RESP=$(curl -s -w "\n%{http_code}" -X POST "$BASE_URL/runs/stream" -H 'Content-Type: application/json' \
    -d "{\"agent_id\":\"echo_agent\",\"input\":{\"messages\":[{\"role\":\"user\",\"content\":\"stateless stream\"}]}}")
BODY=$(echo "$RESP" | head -n -1)
STATUS=$(echo "$RESP" | tail -1)
assert_status "POST /runs/stream" 200 "$STATUS"

echo

# ==========================================================================
bold "6. BACKGROUND RUNS (STATELESS)"
# ==========================================================================

# POST /runs -- create background run
RESP=$(curl -s -w "\n%{http_code}" -X POST "$BASE_URL/runs" -H 'Content-Type: application/json' \
    -d "{\"agent_id\":\"echo_agent\",\"input\":{\"messages\":[{\"role\":\"user\",\"content\":\"background\"}]}}")
BODY=$(echo "$RESP" | head -n -1)
STATUS=$(echo "$RESP" | tail -1)
assert_status "POST /runs (background)" 200 "$STATUS"
BG_RID=$(echo "$BODY" | jq -r '.run_id')
assert_not_empty "bg run_id" "$BODY" '.run_id'

sleep 1

# GET /runs/{run_id}
RESP=$(curl -s -w "\n%{http_code}" "$BASE_URL/runs/$BG_RID")
BODY=$(echo "$RESP" | head -n -1)
STATUS=$(echo "$RESP" | tail -1)
assert_status "GET /runs/{rid}" 200 "$STATUS"
assert_json "bg run status" "$BODY" '.status' "success"

# GET /runs/{run_id}/wait
RESP=$(curl -s -w "\n%{http_code}" "$BASE_URL/runs/$BG_RID/wait")
BODY=$(echo "$RESP" | head -n -1)
STATUS=$(echo "$RESP" | tail -1)
assert_status "GET /runs/{rid}/wait" 200 "$STATUS"
assert_not_empty "wait has values" "$BODY" '.values'

# POST /runs/{run_id}/cancel
RESP2=$(curl -s -X POST "$BASE_URL/runs" -H 'Content-Type: application/json' \
    -d "{\"agent_id\":\"echo_agent\",\"input\":{\"messages\":[{\"role\":\"user\",\"content\":\"to cancel\"}]}}")
CANCEL_RID2=$(echo "$RESP2" | jq -r '.run_id')
RESP=$(curl -s -w "\n%{http_code}" -X POST "$BASE_URL/runs/$CANCEL_RID2/cancel" -H 'Content-Type: application/json')
BODY=$(echo "$RESP" | head -n -1)
STATUS=$(echo "$RESP" | tail -1)
assert_status "POST /runs/{rid}/cancel" 200 "$STATUS"

# DELETE /runs/{run_id}
sleep 0.5
RESP=$(curl -s -w "\n%{http_code}" -X DELETE "$BASE_URL/runs/$BG_RID")
STATUS=$(echo "$RESP" | tail -1)
assert_status "DELETE /runs/{rid}" 200 "$STATUS"

# POST /runs/search
RESP=$(curl -s -w "\n%{http_code}" -X POST "$BASE_URL/runs/search" -H 'Content-Type: application/json' -d '{}')
BODY=$(echo "$RESP" | head -n -1)
STATUS=$(echo "$RESP" | tail -1)
assert_status "POST /runs/search" 200 "$STATUS"

echo

# ==========================================================================
bold "7. STORE"
# ==========================================================================

# PUT /store/items
STATUS=$(curl -s -o /dev/null -w "%{http_code}" -X PUT "$BASE_URL/store/items" -H 'Content-Type: application/json' \
    -d '{"namespace":["curl","test"],"key":"mykey","value":{"data":"hello"}}')
assert_status "PUT /store/items" 200 "$STATUS"

# GET /store/items
RESP=$(curl -s -w "\n%{http_code}" "$BASE_URL/store/items?namespace=curl&namespace=test&key=mykey")
BODY=$(echo "$RESP" | head -n -1)
STATUS=$(echo "$RESP" | tail -1)
assert_status "GET /store/items" 200 "$STATUS"
assert_json "store value" "$BODY" '.value.data' "hello"

# POST /store/items/search
RESP=$(curl -s -w "\n%{http_code}" -X POST "$BASE_URL/store/items/search" -H 'Content-Type: application/json' \
    -d '{"namespace_prefix":["curl"]}')
BODY=$(echo "$RESP" | head -n -1)
STATUS=$(echo "$RESP" | tail -1)
assert_status "POST /store/items/search" 200 "$STATUS"
assert_json "search has items" "$BODY" '.items | length >= 1' "true"

# POST /store/namespaces
RESP=$(curl -s -w "\n%{http_code}" -X POST "$BASE_URL/store/namespaces" -H 'Content-Type: application/json' -d '{}')
BODY=$(echo "$RESP" | head -n -1)
STATUS=$(echo "$RESP" | tail -1)
assert_status "POST /store/namespaces" 200 "$STATUS"

# DELETE /store/items
STATUS=$(curl -s -o /dev/null -w "%{http_code}" -X DELETE "$BASE_URL/store/items" -H 'Content-Type: application/json' \
    -d '{"namespace":["curl","test"],"key":"mykey"}')
assert_status "DELETE /store/items" 200 "$STATUS"

echo

# ==========================================================================
bold "=========================================="
bold "RESULTS"
bold "=========================================="
echo "Total:  $TOTAL"
green "Passed: $PASS"
if [ $FAIL -gt 0 ]; then
    red "Failed: $FAIL"
    exit 1
else
    green "All tests passed!"
fi
