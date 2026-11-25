// Camera bumper for the forearm camera (SO-101 robot, docs/camera_bumper.md). OpenSCAD.
// A cage round the camera body: four corner rails and a square ring at each end. The camera slides in from the lens side and sits 0.5 mm inside the rails.
// These are the dimensions that were simulated (sim/model.py GUARD, see the document); change the camera numbers to the camera you actually buy.
cam_len  = 90;      // camera body, along its long side (D435 / OAK-D Lite class: 90 x 25 x 25 mm)
cam_h    = 25;      // camera body height
cam_d    = 25;      // camera body depth (from the lens plane back)
gap      = 0.5;     // clearance between the camera and the cage
rail_r   = 4;       // rail thickness, radial (this is how far the cage stands proud of the camera: gap + rail_r = 4.5 mm)
rail_t   = 6;       // rail width, tangential (the simulation has 4: a wider rail is stronger against the 40-60 N that the floor pushed with in the falls)
ring_len = 4;       // ring length along the camera
ring_w   = 4;       // ring wall
$fn = 24;

inner_h = cam_h + 2 * gap;  inner_d = cam_d + 2 * gap;           // the opening the camera sits in
outer_h = inner_h + 2 * rail_r;  outer_d = inner_d + 2 * rail_r;
total   = cam_len + 2 * gap + 2 * ring_len;                      // along the camera

module ring(x) {                                                 // a square frame at one end
    translate([x, 0, 0]) difference() {
        cube([ring_len, outer_h, outer_d], center = true);
        cube([ring_len + 1, inner_h + 2 * (rail_r - ring_w), inner_d + 2 * (rail_r - ring_w)], center = true);
    }
}
module rail(sy, sz) {                                            // an L-shaped corner rail, rail_r thick, rail_t long on each face
    translate([0, sy * (inner_h / 2 + rail_r / 2), sz * (inner_d / 2 + rail_r / 2)])  cube([total, rail_r, rail_r], center = true);
    translate([0, sy * (inner_h / 2 + rail_r / 2), sz * (inner_d / 2 - rail_t / 2 + rail_r)]) cube([total, rail_r, rail_t], center = true);
}
union() {
    ring( (cam_len / 2 + gap + ring_len / 2));
    ring(-(cam_len / 2 + gap + ring_len / 2));
    for (sy = [-1, 1]) for (sz = [-1, 1]) rail(sy, sz);
}
// print on its side, PETG or PLA+, 4 perimeters, 40 % infill. About 14 g.
