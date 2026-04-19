def pytest_addoption(parser):
    parser.addoption("--keep-artifacts", action="store_true", default=False, help="Keep e2e temp dir after test")
