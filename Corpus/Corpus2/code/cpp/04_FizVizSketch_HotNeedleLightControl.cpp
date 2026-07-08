#include "HotNeedleLightControl.h"

HotNeedleLightControlClass::HotNeedleLightControlClass(uint8_t background[NEOPIXEL_COUNT][COLOR_BYTES], uint8_t hotNeedleColor[COLOR_BYTES], float highlightMultiplier, bool useHighlight, uint16_t fadeTime, uint8_t framePeriod, Adafruit_NeoPixel *strip) : LightControlClass(framePeriod, strip)
{
  memcpy(this->backgroundColors, background, COLOR_BYTES*NEOPIXEL_COUNT);
  memcpy(this->hotNeedleColor, hotNeedleColor, COLOR_BYTES);
 
  fadeFrames = fadeTime / framePeriod;
  this->useHighlight = useHighlight;
  this->highlightMultiplier = highlightMultiplier;

  this->maximumLedPosition = 0;
  this->minimumLedPosition = NEOPIXEL_COUNT;
}

// Rendering code

void HotNeedleLightControlClass::renderFrame(uint16_t pos, NEEDLE_DIRECTION dir)
{
  // Increment existing counters
  decrementCounters(ledCounters);

  uint16_t needlePosition = pixelFromInputPosition(pos);
  
  // Set current position hot pixel counter to max
  ledCounters[needlePosition] = fadeFrames;

  draw(needlePosition);
}

void HotNeedleLightControlClass::draw(uint16_t needlePosition)
{
    // Calculate display values for each pixel
  for (uint16_t p = 0; p < NEOPIXEL_COUNT; p++)
  {
    float backgroundRatio = (float)(fadeFrames - ledCounters[p]) / fadeFrames;
    float foregroundRatio = 1.0 - backgroundRatio;
    
    for (uint8_t c = 0; c < COLOR_BYTES; c++)
    {
      if (useHighlight)
      {
        // Foreground color is background color * highlight multiplier
        // Make sure we don't wrap past 255
        int bg = backgroundColors[p][c] * highlightMultiplier;
        if (bg > 255)
        {
          bg = 255;
        }
        ledCurrentColors[p][c] = gammaCorrect((foregroundRatio * bg) + (backgroundRatio * backgroundColors[p][c]));
      }
      else
      {
        ledCurrentColors[p][c] = gammaCorrect((foregroundRatio * hotNeedleColor[c]) + (backgroundRatio * backgroundColors[p][c]));
      }
    }
    strip->setPixelColor(p, ledCurrentColors[p][RED],
                            ledCurrentColors[p][GREEN],
                            ledCurrentColors[p][BLUE]);
  }

  if(useMaximum){
    updateMaximum(needlePosition);
    drawMaximum();
  }

  if(useMinimum){
    updateMinimum(needlePosition);
    drawMinimum();
  }

  strip->show();
}

