# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0


from kenning.core.platform import Platform
from kenning.utils.resource_manager import ResourceURI

# This platforms exists in kenning-resource
# platforms.yml but with changed values
mock_platform = """
96b_meerkat96/mcimx7d/m4:
  compilation_flags:
  - -march=armv7e-m
  - -mcpu=cortex-m4
  - -keys=arm_cpu,cpu
  - -device=arm_cpu
  - -model=mcimx7d
  default_optimizer:
  - TFLiteCompiler
  - TVMCompiler
  default_platform: ZephyrPlatform
  display_name: Meerkat96 (mcimx7d/m4)
  flash_size_kb: 64
  ram_size_kb: 128
  uart_log_baudrate: 115200

"""


class TestPlatformsYml:
    def test_kenning_resources_platforms(self):
        """
        Test for loading a platform from kenning
        resources repository.
        """
        platform = Platform("acrn_adl_crb/atom")
        assert platform.ram_size_kb == 8192
        assert platform.uart_log_baudrate == 115200

    def test_default_kenning_platforms(self):
        """
        Test when loading a platform from the default kenning
        platforms repository.
        """
        platform = Platform("max78002evkit/max78002/m4")
        assert platform.ram_size_kb == 64
        assert platform.uart_baudrate == 115200

    def test_kenning_platforms_priority(self, tmp_path):
        """
        Test ame conflict in ``kenning-resources`` and
        default platforms.yml.
        """
        pfile = tmp_path / "test-platforms.yml"
        pfile.write_text(mock_platform)
        platform = Platform(
            "96b_meerkat96/mcimx7d/m4",
            platforms_definitions=[
                ResourceURI("kenning:///platforms/platforms.yml"),
                pfile,
            ],
        )

        # The loaded platform should come from test-platforms.yml
        # The flash_size_kb and ram_size_kb had been changed.
        assert platform.flash_size_kb == 64
        assert platform.ram_size_kb == 128

    def test_nonexistent_platform(self):
        platform = Platform("nonexistent")
        assert not hasattr(platform, "ram_size_kb")
        assert not hasattr(platform, "uart_baudrate")
