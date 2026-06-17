import sys
import os
from skyfield.api import load

def main():
    bsp_path = sys.argv[1]
    t0_unix_utc = float(sys.argv[2])
    duration_days = int(float(sys.argv[3]))
    
    eph = load(bsp_path)
    ts = load.timescale()
    
    jd0 = 2451545.0 + (t0_unix_utc - 946728000.0) / 86400.0
    
    BODY_CONFIG = [
      ('Sun',     'sun'),
      ('Mercury', 'mercury barycenter'),
      ('Venus',   'venus barycenter'),
      ('Earth',   'earth'),
      ('Moon',    'moon'),
      ('Mars',    'mars barycenter'),
      ('Jupiter', 'jupiter barycenter'),
      ('Saturn',  'saturn barycenter'),
      ('Uranus',  'uranus barycenter'),
      ('Neptune', 'neptune barycenter'),
    ]
    
    print(len(BODY_CONFIG))
    
    h_step_days = 0.5
    n_points = int(duration_days / h_step_days) + 2
    
    print(n_points)
    print(h_step_days)
    
    import numpy as np
    jds = jd0 + np.arange(n_points) * h_step_days
    times = ts.tt_jd(jds)
    
    for name, sf_name in BODY_CONFIG:
        print(name)
        pos = eph[sf_name].at(times)
        px, py, pz = pos.position.km
        vx, vy, vz = pos.velocity.km_per_s
        for idx in range(len(jds)):
            print(f'{px[idx]*1000.0:.6e} {py[idx]*1000.0:.6e} {pz[idx]*1000.0:.6e} {vx[idx]*1000.0:.6e} {vy[idx]*1000.0:.6e} {vz[idx]*1000.0:.6e}')

if __name__ == '__main__':
    main()
