"""Render two v8 fixed-validation handover instants with measured pen-contact markers.

Markers are display overlays at archived world-space contact coordinates. They
do not change the policy rollout, collision model, or anatomical pad labels.
"""

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "assets/wuji_hand2_palm_up_human_demo_v8/scene.xml"
SOURCE = ROOT / "reports/wuji_hand2/palm_up_human_demo_v8/pilot_20261017/final_0000996/fixed_validation"
OUTPUT = ROOT / "reports/wuji_hand2/palm_up_human_demo_v9/gate0_20261019"
FRAMES = (51, 127)  # 1.30 s and 3.20 s, high measured distal load in the review windows.
CAMERAS = ((130, 35), (0, 10), (90, 85))
CONTACT_COLORS = {
    "r_index_finger_tip_sensor_frame": (0.05, 0.9, 1.0, 1.0),
    "r_middle_finger_distal": (1.0, 0.0, 0.9, 1.0),
    "r_thumb_distal": (1.0, 1.0, 1.0, 1.0),
}
CONTACT_LABELS = {
    "r_index_finger_tip_sensor_frame": "I sensor",
    "r_middle_finger_distal": "M distal",
    "r_thumb_distal": "T distal",
}


def project_to_pixels(point, scene):
    """Project a 3-D point through the current MuJoCo free camera."""
    camera = scene.camera[0]
    position = np.mean([eye.pos for eye in scene.camera], axis=0)
    forward = np.asarray(camera.forward)
    up = np.asarray(camera.up)
    right = np.cross(forward, up)
    delta = np.asarray(point) - position
    depth = float(delta @ forward)
    if depth <= 0:
        return None
    half_tangent = camera.frustum_top / camera.frustum_near
    x = 320 + float(delta @ right) / depth / half_tangent * 240
    y = 240 - float(delta @ up) / depth / half_tangent * 240
    return x, y


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    trajectory = np.load(SOURCE / "rollout.npz")
    qpos = trajectory["qpos"]
    contacts = json.loads((SOURCE / "contacts.json").read_text())
    if len(qpos) != len(contacts) or float(trajectory["dt"]) != 0.025:
        raise ValueError("Archived trajectory and contact timing do not agree")
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    finger_colors = {
        "thumb": [1, 0.72, 0.18, 1],
        "index_finger": [1, 0.35, 0.25, 1],
        "middle_finger": [0.2, 0.85, 0.7, 1],
    }
    for geom_id in range(model.ngeom):
        name = model.body(model.geom_bodyid[geom_id]).name
        if name == "pen" and not model.geom_contype[geom_id]:
            model.geom_group[geom_id] = 1
        if not name.startswith("r_"):
            continue
        model.geom_rgba[geom_id] = [0.55, 0.58, 0.65, 1]
        for finger, color in finger_colors.items():
            if f"r_{finger}" in name:
                model.geom_rgba[geom_id] = color
    options = mujoco.MjvOption()
    options.geomgroup[2] = 0
    options.geomgroup[3] = 0
    options.sitegroup[:] = 0
    renderer = mujoco.Renderer(model, height=480, width=640)
    metadata = {
        "scope": "Visual overlays of archived v8 first-episode contact points, not a pad classifier or altered rollout",
        "scene_sha256": hashlib.sha256(SCENE.read_bytes()).hexdigest(),
        "trajectory_sha256": hashlib.sha256((SOURCE / "rollout.npz").read_bytes()).hexdigest(),
        "contacts_sha256": hashlib.sha256((SOURCE / "contacts.json").read_bytes()).hexdigest(),
        "marker_radius_m": 0.004,
        "camera_azimuth_elevation_deg": CAMERAS,
        "frames": [],
    }
    for index in FRAMES:
        data.qpos[:] = qpos[index]
        mujoco.mj_forward(model, data)
        selected = [item for item in contacts[index]
                    if item["body"] in CONTACT_COLORS and item["normal_n"] > 0.02]
        if not any(item["body"] == "r_middle_finger_distal" for item in selected):
            raise ValueError(f"Frame {index} no longer has the expected middle distal contact")
        middle = [item["world_position_m"] for item in selected if item["body"] == "r_middle_finger_distal"]
        camera_target = np.mean(middle, axis=0)
        views = []
        callouts = []
        for azimuth, elevation in CAMERAS:
            camera = mujoco.MjvCamera()
            camera.lookat[:] = camera_target
            camera.distance = 0.21
            camera.azimuth = azimuth
            camera.elevation = elevation
            renderer.update_scene(data, camera=camera, scene_option=options)
            for item in selected:
                marker = renderer.scene.geoms[renderer.scene.ngeom]
                mujoco.mjv_initGeom(marker, mujoco.mjtGeom.mjGEOM_SPHERE,
                                    np.array([0.004] * 3), np.array(item["world_position_m"]),
                                    np.eye(3).ravel(), np.array(CONTACT_COLORS[item["body"]]))
                renderer.scene.ngeom += 1
            by_body = {}
            for body in CONTACT_COLORS:
                group = [item for item in selected if item["body"] == body]
                if not group:
                    continue
                total = sum(item["normal_n"] for item in group)
                position = np.average([item["world_position_m"] for item in group], axis=0,
                                      weights=[item["normal_n"] for item in group])
                by_body[body] = {"normal_n": total, "pixel": project_to_pixels(position, renderer.scene)}
            callouts.append(by_body)
            views.append(renderer.render().copy())
        picture = Image.new("RGB", (1920, 560), "black")
        picture.paste(Image.fromarray(np.concatenate(views, axis=1)), (0, 80))
        draw = ImageDraw.Draw(picture)
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 19)
        time = (index + 1) * 0.025
        draw.text((16, 8), f"Wuji v8 selected rollout | t={time:.2f}s | measured contacts", fill="white", font=font)
        legend_x = 16
        for body, label in CONTACT_LABELS.items():
            group = [item for item in selected if item["body"] == body]
            if not group:
                continue
            normal = sum(item["normal_n"] for item in group)
            color = tuple(int(channel * 255) for channel in CONTACT_COLORS[body][:3])
            item_text = f"{label}: {normal:.2f} N   "
            draw.text((legend_x, 37), item_text, fill=color, font=font)
            legend_x += int(draw.textlength(item_text, font=font))
        draw.text((1130, 37), "Rings mark projected 3D contacts; occlusion is possible.", fill="white", font=font)
        for view_index, by_body in enumerate(callouts):
            for body, point in by_body.items():
                if point["pixel"] is None:
                    continue
                x, y = point["pixel"]
                x += view_index * 640
                y += 80
                color = tuple(int(channel * 255) for channel in CONTACT_COLORS[body][:3])
                draw.ellipse((x - 12, y - 12, x + 12, y + 12), outline=color, width=4)
        path = OUTPUT / f"v8_handover_contact_t{time:.2f}.png"
        picture.save(path)
        metadata["frames"].append({
            "index": index,
            "seconds": time,
            "output": str(path.relative_to(ROOT)),
            "contacts": [{"body": item["body"], "normal_n": item["normal_n"],
                          "world_position_m": item["world_position_m"]} for item in selected],
            "projected_callouts_by_view": callouts,
        })
    renderer.close()
    (OUTPUT / "contact_closeup_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
