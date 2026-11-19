import os
import csv
import numpy as np


def _complementary_layout(planes, satellites_per_plane, shift_step):
    """Fixed 50/30/20 tiers with complementary along-track service points."""
    pattern = [50, 50, 75, 50, 75, 100, 75, 100, 50, 50]
    total = planes * satellites_per_plane
    counts = [round(total * 0.5), round(total * 0.3)]
    counts.append(total - sum(counts))
    # Resolve partial-period quota ties by descending plane, then slot order.
    order = sorted(range(total), key=lambda index: (
        pattern[(index % satellites_per_plane - (index // satellites_per_plane) % 2) % len(pattern)],
        -(index // satellites_per_plane),
    ))
    abilities = np.empty(total, dtype=int)
    cursor = 0
    for ability, count in zip((50, 75, 100), counts):
        abilities[order[cursor:cursor + count]] = ability
        cursor += count
    abilities = abilities.reshape(planes, satellites_per_plane)
    names = [f"P{plane + 1:02d}" for plane in range(planes)]
    # Store each validated allocation before the existing orbit-shift step.
    templates = {
        name: np.roll(abilities[plane], -shift_step * plane).tolist()
        for plane, name in enumerate(names)
    }
    return {
        "plane_templates": names, "templates": templates,
        "shift_step": shift_step,
        "expected_counts": dict(zip((50, 75, 100), counts)),
    }


# Each template is indexed by satellite number; positive shifts use
# base[(i - shift) % satellites_per_plane]. Iridium keeps the separately
# validated dispersed service-point allocation instead of the ten-node motif.
CONSTELLATION_LAYOUTS = {
    (12, 24): _complementary_layout(12, 24, 5),
    (6, 11): {
        "plane_templates": ["P01", "P02", "P03", "P04", "P05", "P06"],
        "templates": {
            "P01": [50, 50, 50, 75, 50, 50, 75, 75, 75, 50, 50],
            "P02": [75, 100, 50, 50, 75, 100, 75, 75, 50, 100, 100],
            "P03": [50, 50, 50, 100, 75, 50, 50, 50, 100, 75, 75],
            "P04": [50, 100, 50, 50, 50, 100, 75, 75, 75, 100, 100],
            "P05": [50, 50, 100, 100, 75, 75, 75, 75, 50, 50, 50],
            "P06": [50, 100, 50, 50, 50, 75, 50, 75, 50, 50, 50],
        },
        "shift_step": 3,
        "expected_counts": {50: 33, 75: 20, 100: 13},
    },
    (27, 13): _complementary_layout(27, 13, 4),
    (18, 36): _complementary_layout(18, 36, 5),
    (36, 20): _complementary_layout(36, 20, 7),
}


class TLEGenerator:
    def __init__(self, output_file, planes, satellites_per_plane, inclo, altitude,
                 argpo_init=90, nodeo_phase=None, argpo_phase=True):
        self.output_file = output_file
        self.satellites_per_plane = satellites_per_plane
        self.planes = planes
        self.inclo = inclo
        self.altitude = altitude
        self.mean_motion = self.mean_motion(altitude)
        self.ecco = "0002000"
        self.bstar = "12222-5"
        self.argpo_init=argpo_init
        self.nodeo_phase=nodeo_phase
        self.argpo_phase=argpo_phase
        self.computing_ability_file = f"{os.path.splitext(output_file)[0]}_computing_ability.csv"
        self.computing_allocation = {}

    def mean_motion(self,altitude):
        earth_radius = 6371.0
        gravitational_constant = 398600.4418
        semi_major_axis = earth_radius + altitude
        period = 2 * np.pi * np.sqrt((semi_major_axis ** 3) / gravitational_constant)
        period_days = period / (60 * 60 * 24)
        mean_motion = 1 / period_days
        return mean_motion

    def tle_checksum(self, line):
        checksum = 0
        for c in line[:-1]:
            if c.isdigit():
                checksum += int(c)
            if c == '-':
                checksum += 1
        return checksum % 10

    def generate_computing_abilities(self):
        """Create the validated 50/75/100 GFLOPS resource layout."""
        layout_key = (self.planes, self.satellites_per_plane)
        layout = CONSTELLATION_LAYOUTS.get(layout_key)
        if layout is None:
            raise ValueError(
                "No configured ability layout for constellation size "
                f"{self.planes}x{self.satellites_per_plane}"
            )

        if len(layout["plane_templates"]) != self.planes:
            raise RuntimeError(
                f"Expected {self.planes} plane templates, got "
                f"{len(layout['plane_templates'])}"
            )

        abilities_by_plane = []
        total_abilities = []
        for plane, template_name in enumerate(layout["plane_templates"], start=1):
            base = layout["templates"][template_name]
            if len(base) != self.satellites_per_plane:
                raise RuntimeError(
                    f"Template {template_name} has {len(base)} satellites, "
                    f"expected {self.satellites_per_plane}"
                )
            shift = (layout["shift_step"] * (plane - 1)) % self.satellites_per_plane
            abilities = [
                base[(index - shift) % self.satellites_per_plane]
                for index in range(self.satellites_per_plane)
            ]
            abilities_by_plane.append(abilities)
            total_abilities.extend(abilities)
            for satellite, ability in enumerate(abilities, start=1):
                sat_name = f"Satellite_{self.altitude}_{plane}_{satellite}"
                self.computing_allocation[sat_name] = {
                    "ability": ability,
                    "template": template_name,
                    "orbit_shift": shift,
                }

        actual_counts = {
            ability: total_abilities.count(ability)
            for ability in (50, 75, 100)
        }
        expected_counts = layout["expected_counts"]
        if actual_counts != expected_counts:
            raise RuntimeError(
                f"Ability allocation mismatch: expected {expected_counts}, "
                f"got {actual_counts}"
            )
        print(
            f"Computing ability allocation for {self.output_file}: "
            f"50={actual_counts[50]}, 75={actual_counts[75]}, "
            f"100={actual_counts[100]}"
        )
        return abilities_by_plane

    def generate_tles(self):
        abilities_by_plane = self.generate_computing_abilities()
        with open(self.output_file, "w") as f, open(self.computing_ability_file, "w", newline="") as computing_file:
            writer = csv.writer(computing_file)
            writer.writerow([
                "satellite_name", "orbit_plane", "satellite_number",
                "computing_ability_gflops", "plane_template", "orbit_shift",
            ])
            for i in range(self.planes):
                nodeo = (180 if self.inclo>=75 else 360) / self.planes * i + (0 if not self.nodeo_phase else 360/ self.planes/self.nodeo_phase[1]*self.nodeo_phase[0])
                computing_abilities = abilities_by_plane[i]
                for j in range(self.satellites_per_plane):
                    sat_name = f"Satellite_{self.altitude}_{i+1}_{j+1}"
                    f.write(sat_name + "\n")
                    line1 = "1 {0:05d}U 00000A   23121.00000000  .00000000  00000+0  {1} 0  999".format(i * 100 + j + 1, self.bstar)
                    line1 += str(self.tle_checksum(line1 + " ")) + "\n"
                    f.write(line1)
                    mo = 360.0 / self.satellites_per_plane * j
                    argpo = self.argpo_init  if i%2==0 or not self.argpo_phase else self.argpo_init-180/self.satellites_per_plane
                    line2 = "2 {0:05d} {1:8.4f} {2:8.4f} {3} {4:8.4f} {5:8.4f} {6:11.8f}  999".format(i * 100 + j + 1, self.inclo, nodeo, self.ecco, argpo, mo, self.mean_motion)
                    line2 += str(self.tle_checksum(line2 + " ")) + "\n"
                    f.write(line2)
                    allocation = self.computing_allocation[sat_name]
                    writer.writerow([
                        sat_name,
                        i + 1,
                        j + 1,
                        f"{computing_abilities[j]:.6f}",
                        allocation["template"],
                        allocation["orbit_shift"],
                    ])
        print(f"TLEs written to {os.path.abspath(self.output_file)}")
        print(f"Computing abilities written to {os.path.abspath(self.computing_ability_file)}")

if __name__ == "__main__":
    constellations = [
        ("wallker", "Satellite_Data/wallker_tles.txt", 12, 24, 60, 500),
        ("Iridium", "Satellite_Data/Iridium_tles.txt", 6, 11, 86.4, 780),
        ("Telesat", "Satellite_Data/Telesat_tles.txt", 27, 13, 98.98, 1015),
        ("OneWeb", "Satellite_Data/OneWeb_tles.txt", 18, 36, 87.9, 1200),
        ("Starlink", "Satellite_Data/Starlink_tles.txt", 36, 20, 70, 570),
    ]

    for name, output_file, planes, satellites_per_plane, inclo, altitude in constellations:
        print(f"Generating {name} constellation...")
        tle_gen = TLEGenerator(
            output_file,
            planes,
            satellites_per_plane,
            inclo,
            altitude,
            argpo_init=90,
            nodeo_phase=None,
            argpo_phase=True,
        )
        tle_gen.generate_tles()
