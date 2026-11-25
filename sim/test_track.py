"""A track that is not seen for a few frames is carried along at its velocity, each frame by that frame's step (it used to be pushed by the time since the last sighting again every frame, so a
missed track drifted away faster and faster), and its velocity is measured from the last place it was actually seen.   python test_track.py"""
import people as P

k = P.Track(0.0, 0.0, 0.0, 1.5, False, 0.4)
k.update(0.2, 0.2, 0.0, 1.5, False, 0.4)                       # moved 0.2 m in 0.2 s: raw velocity 1.0, smoothed with the old 0 -> 0.5 m/s
assert abs(k.vx - 0.5) < 1e-9 and abs(k.x - 0.2) < 1e-9
for i, t in enumerate((0.4, 0.6, 0.8, 1.0, 1.2), 1):          # five frames with nothing seen: the branch of PersonDetector._associate
    if t - k.t < 1.5: k.x, k.y = k.predict(t); k.t_adv = t
    assert abs(k.x - (0.2 + 0.5 * (t - 0.2))) < 1e-9, (i, k.x)   # straight on at 0.5 m/s, not faster
k.update(1.6, 0.9, 0.0, 1.5, False, 0.4)                       # seen again where it is: velocity from (0.2 -> 0.9 over 1.4 s) = 0.5, not from the carried-along position
assert abs(k.vx - 0.5) < 1e-9 and abs(k.x - 0.9) < 1e-9
print('ok')
