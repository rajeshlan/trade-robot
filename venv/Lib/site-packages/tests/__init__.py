import os
import xmlrunner
from tests import config_provider_tests


def run_tests():
    output = 'test-reports/unit/app_config'

    suites = [
        config_provider_tests
    ]

    for suite in suites:
        xmlrunner.XMLTestRunner(output=output).run(suite.get_suite())
        os.system('cls' if os.name == 'nt' else 'clear')
