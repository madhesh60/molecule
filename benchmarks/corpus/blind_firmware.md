# Kestrel Weather Station — Firmware Notes

Working notes for the firmware that runs on the Kestrel remote weather
station. The station sits on a pole in a field for years at a time, so most of
these rules exist because something once went wrong in a place nobody could
reach quickly.

## Hardware

### Board revisions

There are three board revisions in the field. Rev B has the BME280 pressure
sensor on I2C address 0x76; rev C moved it to 0x77 to make room for the light
sensor. Rev D replaced the linear regulator with a buck converter and is the
only revision that supports the solar charge controller. The firmware reads
the revision from three strapping pins at boot and must never assume one.

### Sensors

The station measures air temperature, humidity and pressure (BME280), wind
speed (a reed-switch anemometer that closes once per rotation), wind direction
(a potentiometer vane read on ADC channel 2) and rainfall (a tipping bucket
that tips every 0.2 mm). Wind speed is computed from pulse counts over a
ten-second window. The light sensor is optional and missing on rev B.

### Power budget

The station runs from a 6 Ah lithium iron phosphate pack. Average current must
stay below 1.8 mA for the pack to last through a dark winter month. The
microcontroller sleeps in stop mode between samples and wakes on the real-time
clock alarm. The radio is the biggest consumer: a transmit burst draws about
120 mA for 400 ms, which is why uploads are batched.

### Enclosure and connectors

The enclosure is rated IP66. Every cable enters from below through a gland so
water cannot run along it into the box. The debug connector is a six-pin
Tag-Connect footprint with no header soldered, because a header was once the
path that let condensation short the reset line. Replace the desiccant bag at
every site visit.

## Building and flashing

### Toolchain

The firmware is built with arm-none-eabi-gcc 13 and CMake. The exact compiler
version is pinned in the dev container; do not build releases with whatever
compiler your laptop has, because code size differs between versions and the
image has to fit in 192 KB. Run `make size` after every change that adds a
dependency.

### Flashing a board on the bench

Connect the debug probe and run:

```bash
# Erase, program and verify in one step
make flash PROBE=stlink

# Open the serial console at 115200 baud
make console
```

A blank board needs the bootloader first: `make flash-bootloader`. If the
probe cannot connect, hold the reset button while starting the flash command,
then release it when the tool says it is waiting for the target.

### Bootloader

The bootloader occupies the first 16 KB of flash and is never updated in the
field. It keeps two application slots, A and B, and boots the slot marked
good with the higher sequence number. A new image is only marked good after
it has run for ten minutes and completed one successful upload. If that does
not happen, the next reset falls back to the other slot.

### Over-the-air updates

Updates are delivered over the radio link in 200-byte blocks, at most one
image per station per week to protect the battery. Each image is signed with
Ed25519 and the signature is checked before the slot is marked bootable. An
update is refused when the pack voltage is under 12.4 V, because a brown-out
in the middle of writing flash is the one failure the bootloader cannot fix.

## Runtime behaviour

### Sampling schedule

Sensors are sampled once a minute. Wind is the exception: the anemometer
pulses are counted continuously by a low-power timer and summarised every ten
seconds so gusts are not missed. Samples are averaged into five-minute records
before storage. During a storm, defined as wind above 15 m/s for one minute,
the station switches to one-minute records until the wind drops.

### Local storage

Records are written to an external 8 MB SPI flash chip as a ring buffer, which
holds roughly 40 days of five-minute records. Each record carries a CRC-16 so
a torn write is detected and skipped on read. The write pointer is stored
twice, in two different sectors, and the copy with the valid checksum and the
higher counter wins after a power loss.

### Radio link

The station uses LoRa at 868 MHz with spreading factor 9 by default. It
uploads every 30 minutes and listens for a reply for two seconds afterwards;
that reply window is the only time the station can receive commands. If three
uploads in a row get no acknowledgement, the station steps up to spreading
factor 11 to gain range, and steps back down after a day of clean links.

### Time keeping

The real-time clock drifts by up to two seconds a day with temperature. The
gateway includes the current time in every acknowledgement and the station
corrects its clock when the difference exceeds five seconds. All timestamps
are stored in UTC as seconds since 2020-01-01; local time is a display
concern for the dashboard, never for the firmware.

### Watchdog

The independent watchdog is set to 26 seconds and is fed from exactly one
place: the end of the main loop. Never feed it from an interrupt or a timer,
because that keeps a hung main loop alive forever. Each reset records its
cause in a small log in backup memory, and the cause is sent with the next
upload so repeated watchdog resets show up on the dashboard.

### Low battery behaviour

Below 12.0 V the station stops listening for commands and uploads once every
three hours. Below 11.6 V it stops uploading and only logs locally. Below
11.2 V it stops sampling and sleeps until the solar panel has lifted the pack
back above 12.2 V. The thresholds have hysteresis so the station does not flap
between modes at dusk.

## Data

### Record format

A record is 24 bytes: a four-byte timestamp, then temperature, humidity,
pressure, wind speed, gust, direction and rainfall as scaled integers, a
status byte and the CRC. Temperature is stored in hundredths of a degree
Celsius as a signed 16-bit value. Never change the layout of an existing
field; add a new record version and bump the first status bit instead.

### Calibration

Every anemometer is calibrated in the wind tunnel before it ships and its
slope and offset are stored in the station's configuration block. The
pressure sensor is corrected to sea level on the server, not on the device,
because the station does not know its own altitude reliably. Rain gauges are
checked each spring by pouring exactly 100 ml through the funnel and counting
the tips.

### Configuration block

Settings live in a 256-byte block in internal flash, protected by a CRC-32. A
second copy is kept as a backup and restored automatically when the first
fails its check. The block holds the station identifier, radio keys,
calibration constants and the upload interval. Changing it over the air
requires a command signed by the fleet key.

## Quality

### Unit tests

Logic that does not touch hardware is compiled for the host and tested with
Unity. Drivers are tested against fakes that replay recorded bus traffic. The
host test suite runs in under 20 seconds and is required to pass before a
merge. Aim the tests at the conversions and state machines; that is where the
field bugs have come from.

### Hardware-in-the-loop rig

A rack in the lab holds one board of each revision wired to signal generators
that imitate the sensors. The rig runs every release candidate for 72 hours,
including simulated storms, radio loss and brown-outs. A candidate that
resets even once without an injected fault is rejected.

### Static analysis

Every build runs cppcheck and the compiler with warnings as errors. The code
follows a reduced MISRA C subset: no dynamic allocation after start-up, no
recursion, and every switch has a default case. Deviations need a comment
that names the rule and the reason.

### Release checklist

A release needs: a green hardware-in-the-loop run, the image size recorded in
the changelog, a signed image uploaded to the fleet server, and a staged
rollout plan. Tag the commit as `fw-MAJOR.MINOR.PATCH`. The first field
installs always go to the five stations nearest the workshop.

### Staged rollout

A new firmware version goes to five nearby stations first, then to ten
percent of the fleet after one week without resets, then to everyone after a
second week. Rollout pauses automatically if more than two stations in a
stage report a watchdog reset. Stations on rev B boards are always updated
last because they cannot be recovered without a site visit.

## In the field

### Site visit checklist

Bring the debug probe, a charged spare pack, desiccant, a multimeter and the
torque driver for the mast clamps. Photograph the installation before you
touch anything. Check the anemometer spins freely, clear the rain gauge
funnel, and download the local storage if the station has been offline. Log
the visit in the fleet tracker before you leave the site.

### Diagnosing a silent station

When a station stops reporting, look at its last uploads first: falling pack
voltage points to a power problem, rising missed acknowledgements point to
the radio link, and a string of resets points to firmware. If the last upload
looked healthy, the usual cause is physical: a broken antenna cable or a
flooded enclosure. Do not push an update to a silent station to "wake it up".

### Lightning and surge damage

Stations on exposed ridges take surges through the anemometer cable. After a
strike the usual symptom is a wind speed stuck at zero while everything else
works. The input protection board is designed to be the part that dies:
replace it, do not bypass it. Boards with a damaged analogue input should be
returned to the workshop rather than repaired in the field.

### Decommissioning

When a station is retired, erase its radio keys with the `factory-wipe`
command before it leaves the site, remove it from the fleet tracker, and
recycle the battery pack through the approved handler. Keep the last 40 days
of records from local storage, because the server copy may be incomplete if
the link was already failing.
