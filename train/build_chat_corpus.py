"""Build you/bot rocket fact text for the 1M-param char-RNN.

Hand facts + Kaggle numbers. No 3k-param baby talk.
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path

VOCAB = "abcdefghijklmnopqrstuvwxyz 0123456789.,:?!'-\n"
HERE = Path(__file__).parent
DATA = HERE / "data"
KAGGLE_SLUG = "sergionefedov/space-missions-and-7500-launches-1957-2024"

HAND_QA = [
    ("what does the rocket do", "it throws mass out the nozzle. thrust rises and the pad is left behind."),
    ("how does a rocket fly", "momentum. hot gas goes one way, the vehicle goes the other. that is newton."),
    ("why does a rocket need an engine", "the engine is a controlled fire. it turns fuel and oxidizer into high speed exhaust."),
    ("what is thrust", "thrust is the force from the exhaust. if thrust beats weight, the rocket lifts."),
    ("what is isp", "isp is specific impulse. it is how many seconds of thrust you get per unit of propellant."),
    ("what is a nozzle", "the nozzle expands the exhaust. pressure turns into speed. that is where most thrust is made."),
    ("what is a combustion chamber", "fuel and oxidizer mix and burn there. chamber pressure feeds the nozzle."),
    ("what is an oxidizer", "a rocket in vacuum has no air. it must carry oxygen or another oxidizer."),
    ("what is rp-1", "rp-1 is refined kerosene. saturn v and falcon 9 burn rp-1 with liquid oxygen."),
    ("what is lox", "lox is liquid oxygen. it is the oxidizer on most kerolox and hydrolox engines."),
    ("what is hydrolox", "liquid hydrogen plus liquid oxygen. high isp, bulky tanks. rs-25 and j-2 used it."),
    ("what is methalox", "liquid methane plus liquid oxygen. raptor and be-4 burn that mix."),
    ("what is a gas generator", "a small burn drives the turbopumps. the rest of the propellant goes to the main chamber."),
    ("what is staged combustion", "turbine gas is dumped into the main chamber, not wasted overboard. higher pressure, harder plumbing."),
    ("what is a turbopump", "it shoves propellant into the chamber at high pressure. without it the chamber starves."),
    ("what is a first stage", "the heavy booster. it lifts off the pad, then drops so the upper stage can finish the job."),
    ("what is a second stage", "a lighter engine and tank. it lights after staging and pushes the payload to orbit."),
    ("what is staging", "you drop empty tanks and engines. mass falls, acceleration rises."),
    ("what is leo", "low earth orbit. about 200 to 2000 km up. you need about 7.8 km per s plus losses."),
    ("what is gto", "geostationary transfer orbit. an upper stage leaves you on an ellipse to geo."),
    ("what is escape velocity", "from earth about 11.2 km per s. a moon stack never holds that in one burn from the pad."),
    ("how fast is orbit", "leo is about 7.8 km per s. most of a rocket's job is horizontal speed, not height."),
    ("why do rockets turn", "a gravity turn. you pitch over so the velocity goes sideways into orbit, not straight up."),
    ("what is max-q", "peak dynamic pressure. the air is still thick and the speed is already high. the vehicle is most loaded."),
    ("what is a hold-down", "clamps keep the stack on the pad until engines are at full thrust, then they release."),
    ("what is ullage", "a small push so liquid sits on the pump inlets in free fall before an upper-stage burn."),
    ("what is saturn v", "nasa moon rocket. 3 stages. 5 f-1 engines on the first stage. it put apollo on a path to the moon."),
    ("how many engines did saturn v have", "5 f-1 engines on s-ic, 5 j-2 on s-ii, 1 j-2 on s-ivb."),
    ("what is the f-1", "the f-1 is a kerolox gas-generator engine. about 6.8 million newtons of thrust. saturn v used 5."),
    ("what did the f-1 burn", "the f-1 burned rp-1 and liquid oxygen."),
    ("how much thrust did the f-1 have", "about 6.8 mn each. five of them held the first stage at lift-off."),
    ("what is the j-2", "a hydrolox upper-stage engine. s-ii had five. s-ivb had one and could restart."),
    ("what rocket went to the moon", "saturn v. apollo 8 was the first crew around the moon. apollo 11 landed."),
    ("what is falcon 9", "a two-stage spacex rocket. 9 merlin engines on the first stage. the booster can land."),
    ("does falcon 9 land", "yes. after stage sep the first stage boosts back and lands on a pad or a ship."),
    ("what is merlin", "merlin is spacex's kerolox engine. gas generator, pintle injector. 9 of them on falcon 9."),
    ("what does merlin burn", "merlin burns rp-1 and liquid oxygen."),
    ("what is falcon heavy", "three falcon 9 cores strapped together. side boosters can land. center core is harder."),
    ("what is raptor", "spacex methalox staged-combustion engine. it is the engine on starship and super heavy."),
    ("what does raptor burn", "raptor burns liquid methane and liquid oxygen."),
    ("what is starship", "a stainless steel stack. super heavy booster plus ship. methalox, many raptors, meant to be fully reused."),
    ("what is sls", "nasa's space launch system. solid boosters plus rs-25 cores leftover from shuttle."),
    ("what is the rs-25", "the space shuttle main engine. hydrolox, staged combustion. now it flies on sls."),
    ("what did the shuttle burn", "two solid boosters plus 3 rs-25 engines on liquid hydrogen and liquid oxygen."),
    ("what is a solid rocket", "the grain is the fuel and oxidizer already mixed. simple, high thrust, you cannot throttle off cleanly."),
    ("what is a liquid rocket", "tanks, pumps, valves. you can throttle, shut down, and often restart."),
    ("what is soyuz", "the russian crew workhorse. an r-7 family stack. it still flies people to orbit."),
    ("what is proton", "a heavy soviet-era storable-propellant rocket. hypergolic, toxic, high payload for its day."),
    ("what is ariane 5", "esa heavy lift. it flew atms, probes, and big comsats from french guiana."),
    ("what is electron", "rocket lab's small orbital rocket. electric pumps, carbon tanks, launches from new zealand."),
    ("what is new glenn", "blue origin's heavy orbital rocket. be-4 methalox engines, reusable first stage."),
    ("what is the be-4", "blue origin methalox engine. new glenn and vulcan use it."),
    ("what is atlas v", "ula rocket. rd-180 on the first stage. it flew many us national security loads."),
    ("what is the rd-180", "a russian kerolox staged-combustion engine. two chambers, one turbine set. flew on atlas v."),
    ("what is pslv", "india's polar satellite launch vehicle. it is known for cheap, reliable small-sat rides."),
    ("what is long march", "china's launch family. many variants, from crew to heavy geo stacks."),
    ("where is baikonur", "kazakhstan. soviet and russian crews have launched there since sputnik."),
    ("where is the cape", "florida. cape canaveral and kennedy space center lc-39 sit on the east coast for eastward launches."),
    ("why launch near the equator", "earth already spins east. you steal that speed for geo and gto."),
    ("why launch from vandenberg", "south and polar orbits. you do not overfly cities on the west coast."),
    ("what is a launch window", "the times the earth, the pad, and the target line up. miss it and you wait."),
    ("what is a payload fairing", "the nose cone. it shields the satellite from air, then it is jettisoned."),
    ("what is a payload adapter", "the ring that bolts the satellite to the upper stage."),
    ("what is delta-v", "change in velocity. tanks and isp buy you delta-v. orbit and landing spend it."),
    ("what is mass fraction", "how much of the stack is propellant. rockets are mostly tank."),
    ("why are rocket tanks thin", "they are pressure vessels. extra metal is dead weight you must accelerate."),
    ("what is film cooling", "a layer of propellant on the wall so the chamber does not melt."),
    ("what is regenerative cooling", "cold propellant runs in the jacket before it burns. the wall gives heat to the fuel."),
    ("what is a pintle injector", "one stream hits another. merlin uses it. it is stable and simple."),
    ("what is combustion instability", "the chamber rings like a bell. f-1 needed baffles so the 5 engines did not tear themselves apart."),
    ("what is a start sequence", "pumps spin, valves open in order, igniters fire. a bad sequence kills the engine."),
    ("what is an abort", "you stop the count or leave the stack. on a crew rocket that can mean an escape tower."),
    ("what is a range safety", "if the vehicle turns toward land, the range can destroy it."),
    ("what is reentry", "the air turns speed into heat. a booster or a capsule needs a shield or a slow-down burn."),
    ("how does a booster land", "it kills horizontal speed, boosts back, then burns again to kill vertical speed over the pad."),
    ("what is a grid fin", "control surfaces in the air on the way down. falcon 9 uses them to steer the booster."),
    ("what is a droneship", "an uncrewed barge. falcon boosters land there when there is not enough propellant to reach land."),
    ("what is super heavy", "the starship booster. many raptors, catch planned at the tower."),
    ("what is a hot stage", "the upper engine lights before the stages fully separate. it can save time and residual ullage issues."),
    ("what is a vacuum nozzle", "a bigger bell. it expands further because there is no sea-level air to separate the flow."),
    ("why is a sea-level nozzle shorter", "over-expansion in thick air separates the flow and can wreck the bell."),
    ("what is tsiolkovsky", "the rocket equation. delta-v equals ve times ln of mass ratio. that is why staging exists."),
    ("what is the rocket equation", "delta-v = ve * ln(m0/mf). more exhaust speed or more mass ratio, more delta-v."),
    ("why is hydrogen hard", "it is cold, leaks, and needs huge tanks. the isp is worth it on upper stages."),
    ("why is methane used", "easier to store than hydrogen, cleaner than rp-1, good for reuse and mars isru stories."),
    ("what is hypergolic", "two fluids that ignite on contact. no spark. toxic, but they restart well. used on many spacecraft."),
    ("what is a reaction control system", "small thrusters for attitude. not the main engine."),
    ("what is gimbals", "the engine swivels. that is how most big rockets steer in the thick air."),
    ("what is a roll program", "right after lift-off the stack rolls to the launch azimuth."),
    ("what was sputnik", "1957. an r-7 put the first satellite in orbit. the space age started with a beep."),
    ("what was apollo 11", "july 1969. armstrong and aldrin landed. collins stayed in the command module."),
    ("what was the saturn v first stage", "s-ic. 5 f-1 engines, rp-1 and lox, about 2.5 minutes, then staging."),
    ("how long did saturn v first stage burn", "about 161 seconds. then the s-ii hydrolox stage lit."),
    ("what is a teardrop tank", "common bulkhead or shared tank shapes to cut mass. hydrogen tanks are huge."),
    ("can this chip chat", "no. it is a 1 million parameter character model. it continues rocket facts, not a general assistant."),
    ("are you chatgpt", "no. this is a tiny rnn on an esp32. no cloud, no tokenizer, next-character only."),
    ("hello", "ready. ask about engines, stages, orbit, or a named rocket."),
    ("what can you talk about", "rockets, engines, propellant, staging, orbit, and launch sites."),
]


def clip_vocab(text: str) -> str:
    text = text.lower().replace("\r\n", "\n").replace("\r", "\n")
    text = "".join(ch if ch in VOCAB else " " for ch in text)
    while "  " in text:
        text = text.replace("  ", " ")
    return text.strip()


def pair(user: str, bot: str) -> str:
    return f"you: {clip_vocab(user)}\nbot: {clip_vocab(bot)}"


def name_ok(s: str) -> bool:
    s = clip_vocab(s)
    if len(s) < 2 or len(s) > 40:
        return False
    return bool(re.search(r"[a-z0-9]", s))


def num(s: str) -> str | None:
    try:
        v = float(s)
    except (TypeError, ValueError):
        return None
    if v != v:
        return None
    if abs(v - round(v)) < 1e-6:
        return str(int(round(v)))
    return f"{v:.1f}"


def col(row: dict, *names: str) -> str:
    lower = {k.lower().strip(): v for k, v in row.items() if k}
    for n in names:
        v = lower.get(n.lower())
        if v is None:
            continue
        v = str(v).strip()
        if v and v.lower() not in ("nan", "none", "null", "-"):
            return v
    return ""


def from_rockets_csv(path: Path) -> list[str]:
    lines: list[str] = []
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            name = col(row, "rocket", "rocket_name", "vehicle", "name", "family")
            if not name_ok(name):
                continue
            n = clip_vocab(name)
            country = clip_vocab(col(row, "country", "operator_country", "nation"))
            prop = clip_vocab(col(row, "propellant", "fuel").replace("/", " "))
            reusable = col(row, "reusable", "first_stage_recoverable")
            year = num(col(row, "first_flight"))
            leo = num(col(row, "payload_to_leo_kg"))
            engines = num(col(row, "first_stage_engines"))
            height = num(col(row, "height_m"))
            bits = [f"{n} is an orbital rocket"]
            if country:
                bits.append(f"from {country}")
            if year:
                bits.append(f"first flight {year}")
            lines.append(pair(f"what is {n}", ". ".join(bits) + "."))
            if leo:
                lines.append(pair(f"what payload does {n} have", f"{n} is about {leo} kg to leo."))
            if engines:
                lines.append(pair(f"how many first stage engines on {n}", f"{n} has {engines} first stage engines."))
            if height:
                lines.append(pair(f"how tall is {n}", f"{n} is about {height} m tall."))
            if reusable in ("1", "true", "yes"):
                lines.append(pair(f"is {n} reusable", f"yes. {n} is listed as reusable."))
            elif reusable in ("0", "false", "no"):
                lines.append(pair(f"is {n} reusable", f"no. {n} is expendable in this table."))
            if prop:
                lines.append(pair(f"what does {n} burn", f"{n} burns {prop}."))
            if len(lines) > 500:
                break
    return lines


def from_launches_csv(path: Path) -> list[str]:
    lines: list[str] = []
    with path.open(encoding="utf-8", newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i > 800:
                break
            rocket = col(row, "rocket", "rocket_name", "vehicle", "launcher")
            site = col(row, "site", "launch_site", "location", "pad")
            outcome = col(row, "outcome", "success", "result", "status")
            mission = col(row, "mission", "mission_name", "payload", "name")
            if not name_ok(rocket):
                continue
            n = clip_vocab(rocket)
            year = num(col(row, "year"))
            success = col(row, "launch_success", "success")
            orbit = clip_vocab(col(row, "orbit"))
            if success in ("1", "true", "yes") or "success" in outcome.lower():
                extra = f" in {year}" if year else ""
                dest = f" toward {orbit}" if orbit else ""
                lines.append(pair(f"did {n} launch{extra}", f"yes. {n} launched{extra}{dest}."))
            elif success in ("0", "false", "no"):
                lines.append(pair(f"did {n} fail", f"yes. a {n} launch is marked failed in the table."))
            if name_ok(site) and i % 3 == 0:
                s = clip_vocab(site)
                lines.append(pair(f"where has {n} launched", f"{n} has launched from {s}."))
            if len(lines) > 400:
                break
    return lines


def from_sites_csv(path: Path) -> list[str]:
    lines: list[str] = []
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            site = col(row, "site_name", "site", "name")
            country = clip_vocab(col(row, "country"))
            if not name_ok(site):
                continue
            s = clip_vocab(site)
            lines.append(pair(f"what is {s}", f"{s} is a launch site. orbital stacks fly from there."))
            if country:
                lines.append(pair(f"where is {s}", f"{s} is in {country}."))
            if len(lines) > 80:
                break
    return lines


def csv_search_roots(primary: Path) -> list[Path]:
    roots = [primary]
    home_archive = Path.home() / "Downloads" / "archive"
    try:
        if home_archive.exists() and home_archive.resolve() != primary.resolve():
            roots.append(home_archive)
    except OSError:
        pass
    return roots


def find_csv(data_dir: Path, *names: str) -> Path | None:
    for root in csv_search_roots(data_dir):
        if not root.exists():
            continue
        for name in names:
            p = root / name
            if p.exists():
                return p
            for q in root.rglob("*.csv"):
                if q.name.lower() == name.lower():
                    return q
    return None


def try_kaggle(data_dir: Path) -> None:
    token = Path.home() / ".kaggle" / "kaggle.json"
    if not token.exists():
        print("no ~/.kaggle/kaggle.json — skip download, use local/fallback")
        return
    data_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "kaggle",
        "datasets",
        "download",
        "-d",
        KAGGLE_SLUG,
        "-p",
        str(data_dir),
        "--unzip",
    ]
    print("kaggle:", " ".join(cmd))
    try:
        subprocess.check_call(cmd)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"kaggle download failed: {e}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--out", type=Path, default=HERE / "chat_corpus.txt")
    parser.add_argument(
        "--skip-kaggle",
        action="store_true",
        help="no-op; hand facts are already the default (kept for old command lines)",
    )
    parser.add_argument("--with-kaggle", action="store_true", help="mix CSV rows back in")
    parser.add_argument("--repeat", type=int, default=40)
    args = parser.parse_args()

    if args.with_kaggle and not args.skip_kaggle:
        try_kaggle(args.data)

    core = [pair(u, b) for u, b in HAND_QA]
    blocks: list[str] = []
    if args.with_kaggle:
        print(f"data dir: {args.data}")
        rockets = find_csv(args.data, "rockets.csv")
        launches = find_csv(args.data, "launches.csv")
        sites = find_csv(args.data, "launch_sites.csv")
        if rockets:
            print(f"using {rockets}")
            blocks.extend(from_rockets_csv(rockets))
        if launches:
            print(f"using {launches}")
            blocks.extend(from_launches_csv(launches))
        if sites:
            print(f"using {sites}")
            blocks.extend(from_sites_csv(sites))
    else:
        print("hand facts only — no kaggle csv")

    text = "\n\n".join(core * max(1, args.repeat) + blocks) + "\n"
    args.out.write_text(text, encoding="utf-8")
    print(f"wrote {len(text)} chars, {text.count('you:')} turns -> {args.out}")


if __name__ == "__main__":
    main()
