#!/usr/bin/env python3
"""Summarize paired deltas by scene and camera group."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METRICS = ("psnr", "ssim", "lpips", "dists")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", action="append", type=Path, required=True)
    parser.add_argument("--population", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = {}
    sensor_seed_values: dict[str, dict[str, dict[str, float]]] = {}
    for path in args.result:
        payload = json.loads(path.read_text(encoding="utf-8"))
        population = next(
            item
            for item in payload["populations"]
            if item["population"] == args.population
        )
        seed = path.name.split("_", 1)[0].removeprefix("seed")
        by_scene: dict[str, dict[str, list[float]]] = {}
        scene_sensor: dict[str, str] = {}
        for row in population["records"]:
            scene = str(row["scene"])
            sensor = str(row["sensor"])
            scene_sensor.setdefault(scene, sensor)
            if scene_sensor[scene] != sensor:
                raise ValueError(f"scene {scene} spans multiple sensors")
            for metric in METRICS:
                by_scene.setdefault(scene, {m: [] for m in METRICS})[
                    metric
                ].append(
                    float(row[f"candidate_{metric}"])
                    - float(row[f"base_{metric}"])
                )

        scene_values = {}
        by_sensor: dict[str, dict[str, list[float]]] = {}
        for scene in sorted(by_scene):
            sensor = scene_sensor[scene]
            scene_values[scene] = {
                metric: float(np.mean(by_scene[scene][metric]))
                for metric in METRICS
            }
            by_sensor.setdefault(sensor, {m: [] for m in METRICS})
            for metric in METRICS:
                by_sensor[sensor][metric].append(scene_values[scene][metric])
        seeds[seed] = {
            "scene_values": scene_values,
            "metric_means": {
                metric: float(
                    np.mean(
                        [value[metric] for value in scene_values.values()]
                    )
                )
                for metric in METRICS
            },
            "leave_one_sensor_out": {
                sensor: {
                    "psnr": float(
                        np.mean(
                            [
                                value["psnr"]
                                for scene, value in scene_values.items()
                                if scene_sensor[scene] != sensor
                            ]
                        )
                    ),
                    "scene_count": sum(
                        scene_sensor[scene] != sensor for scene in scene_values
                    ),
                }
                for sensor in sorted(set(scene_sensor.values()))
            },
        }
        sensor_seed_values[seed] = {
            sensor: {
                metric: float(np.mean(values[metric]))
                for metric in METRICS
            }
            for sensor, values in sorted(by_sensor.items())
        }

    sensor_order = sorted(sensor_seed_values[next(iter(seeds))])
    sensors = {}
    for sensor in sensor_order:
        sensors[sensor] = {
            "seed_values": {
                seed: sensor_seed_values[seed][sensor] for seed in sorted(seeds)
            },
            "psnr_positive_seed_count": sum(
                sensor_seed_values[seed][sensor]["psnr"] > 0.0
                for seed in seeds
            ),
        }
    output = {
        "schema_version": 1,
        "result_files": [str(path) for path in args.result],
        "population": args.population,
        "seeds": seeds,
        "sensors": sensors,
        "sensor_psnr_positive_for_all_seeds": sum(
            item["psnr_positive_seed_count"] == len(seeds)
            for item in sensors.values()
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
