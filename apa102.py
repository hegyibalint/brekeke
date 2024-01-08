import spidev


class RgbColor:
    def __init__(self, r: int, g: int, b: int) -> None:
        self.r = r
        self.g = g
        self.b = b


class APA102:
    """
    APA102 LED strip driver

    Implementation based on: https://cdn-shop.adafruit.com/datasheets/APA102.pdf
    """

    def __init__(self, num_led: int) -> None:
        self.colors = [RgbColor(0, 0, 0)] * num_led
        self.spi = spidev.SpiDev()
        self.spi.open(0, 0)
        self.spi.max_speed_hz = 8000000

    def __del__(self) -> None:
        self.spi.close()

    def set_led(self, led: int, r: int, g: int, b: int) -> None:
        self.colors[led] = RgbColor(r, g, b)
        update_colors()

    def update_colors(self) -> None:
        """
        Send the current colors to the LED strip
        """
        # Start frame
        self.spi.xfer([0x00, 0x00, 0x00, 0x00])

        # Send colors
        for color in self.colors:
            self.spi.xfer([0xFF, color.b, color.g, color.r])

        # End frame
        self.spi.xfer([0xFF, 0xFF, 0xFF, 0xFF])
