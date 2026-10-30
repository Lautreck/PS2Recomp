#pragma once

#include <cstdint>

// Top-left host overlay (raylib, after the guest frame is drawn). Not part of
// the latched GS image, so frame shots do not contain it.
// TEKKEN4_HUD=0 starts hidden. F3 toggles. TEKKEN4_HUD_LOG=1 prints one line
// per second to stdout.
namespace ps2_hud
{
    // One host present. newGuestFrame: the presented vsync tick changed (GAME fallback).
    void onHostPresent(bool newGuestFrame);

    // One pass of the game's main loop, from a game override (any thread). Once a game reports frames, GAME
    // counts these: a 30 fps movie shown on every vsync tick still reads 30.
    void noteGameFrame();

    // Publish one completed press: t4 - t1 in milliseconds. Window thread.
    void noteSample(double ms);

    void toggle();

    // Refresh the 2 Hz numbers, optional log, then paint if visible.
    void draw();
}
