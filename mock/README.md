# Mock DA 4Server

`mock_da4_server` is a mock server implemented based on the OpenAPI of Digital Annealer V4 API.

## How to use

`mock_da4_server` can be started using Docker.

1. Build the Docker image
    ```bash
    docker build -t mock_da4_server .
    ```
2. Start the Docker container
    ```bash
    docker run -p 8000:8000 mock_da4_server
    ```
3. View available APIs at `http://0.0.0.0:8000/docs`

> [!IMPORTANT] When using the API, please specify `"mock-api-key"` as the token. Any other token will be treated as an invalid token according to the implementation.

## Specification

`mock_da4_server` specification differs from Digital Annealer V4 in the following aspects:

- It does not perform any optimization calculations. It only assigns random values to the variable numbers included in the `QuboRequest`.
- It only supports authentication via the `X-Api-Key` header.
- It does not tolerate missing `Accept` and `Content-Type` headers.
- Some values that can be specified in `FujitsuDA3Solver` are restricted to ranges where Digital Annealer V4 does not generate numeric errors.
- `BinaryPolynomialTerm` only supports `"c"` and `"p"`. It does not support `"coefficent"` and `"polynomials"`.
