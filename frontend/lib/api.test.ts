import assert from "node:assert/strict";
import test from "node:test";

import { ApiError, parseApiErrorPayload } from "./api.ts";

test("parseApiErrorPayload extracts FastAPI detail codes", () => {
  assert.deepEqual(parseApiErrorPayload(403, '{"detail":"PASSWORD_CHANGE_REQUIRED"}'), {
    status: 403,
    code: "PASSWORD_CHANGE_REQUIRED",
    message: "PASSWORD_CHANGE_REQUIRED",
  });
});

test("parseApiErrorPayload falls back to response text", () => {
  assert.deepEqual(parseApiErrorPayload(500, "server exploded"), {
    status: 500,
    code: null,
    message: "server exploded",
  });
});

test("ApiError keeps HTTP status and code", () => {
  const error = new ApiError(401, "INVALID_CREDENTIALS", "INVALID_CREDENTIALS");
  assert.equal(error.status, 401);
  assert.equal(error.code, "INVALID_CREDENTIALS");
});
