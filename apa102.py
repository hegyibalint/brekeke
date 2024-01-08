from time import sleep, time
from dataclasses import dataclass

import spidev


@dataclass
class RgbbColor:
    r: int
    g: int
    b: int
    brightness: int = 31

    def __post_init__(self) -> None:
        self.r = self._check(self.r, 0, 255, "r")
        self.g = self._check(self.g, 0, 255, "g")
        self.b = self._check(self.b, 0, 255, "b")
        self.brightness = self._check(self.brightness, 0, 31, "brightness")

    def _check(self, value: int, min_value: int, max_value: int, name: str) -> int:
        if value < min_value or max_value < value:
            raise ValueError(f"{name} must be between {min_value} and {max_value}")
        return value


class APA102:
    """
    APA102 LED strip driver

    Implementation based on: https://cdn-shop.adafruit.com/datasheets/APA102.pdf
    """

    def __init__(self, num_led: int) -> None:
        self.colors = [RgbbColor(0, 0, 0)] * num_led
        self.spi = spidev.SpiDev()
        self.spi.open(0, 0)
        self.spi.max_speed_hz = 8000000

    def __del__(self) -> None:
        self.spi.close()

    def __getitem__(self, led: int) -> RgbbColor:
        return self.colors[led]

    def __setitem__(self, led: int, color: RgbbColor) -> None:
        self.colors[led] = color

    def set_all_leds(self, r: int, g: int, b: int, brightness=None) -> None:
        for i in range(len(self.colors)):
            self.set_led(i, r, g, b, brightness)
        self._update_colors()

    def set_led(self, led: int, r: int, g: int, b: int, brightness=None) -> None:
        self.colors[led] = RgbbColor(r, g, b, brightness)
        self._update_colors()

    def _update_colors(self) -> None:
        """
        Send the current colors to the LED strip
        """
        # Start frame
        self.spi.xfer([0x00, 0x00, 0x00, 0x00])

        # Send colors
        for color in self.colors:
            data = [0b11100000 | color.brightness, color.b, color.g, color.r]
            self.spi.xfer(data)

        # End frame
        self.spi.xfer([0xFF, 0xFF, 0xFF, 0xFF])


def test() -> None:
    led_strip = APA102(3)

    while True:
        millis = int(time() * 10)
        # From 0 to 128
        # Phase shifted by colors
        led_strip[0] = RgbbColor(millis % 128, 0, 0, 1)
        led_strip[1] = RgbbColor(0, millis % 128, 0, 1)
        led_strip[2] = RgbbColor(0, 0, millis % 128, 1)
        led_strip._update_colors()
        sleep(0.1)


if __name__ == "__main__":
    test()
