from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Any, Tuple


@dataclass
class DriveState:
    x: float = 0.0
    y: float = 0.0
    yaw_deg: float = 0.0
    speed_mps: float = 0.0
    steering_input: float = 0.0
    steering_input_raw: float = 0.0
    steering_deg: float = 0.0
    slip_deg: float = 0.0
    powersliding: bool = False
    wheel_spin_front_deg: float = 0.0
    wheel_spin_rear_deg: float = 0.0
    front_left_steer_deg: float = 0.0
    front_right_steer_deg: float = 0.0
    motorcycle_speed_ratio: float = 0.0
    motorcycle_lean_input: float = 0.0
    motorcycle_lean_velocity: float = 0.0
    driver_turn_input: float = 0.0
    powerslide_anim_blend: float = 0.0
    braking: bool = False
    brake_intensity: float = 0.0
    # Dynamic preview state. lateral_mps is positive to the vehicle's left;
    # yaw_rate_rps is positive for a left turn. The old Studio assigned an
    # artificial slip angle directly, which made powerslides look like a
    # chassis transform rather than tire dynamics.
    lateral_mps: float = 0.0
    yaw_rate_rps: float = 0.0
    world_vx: float = 0.0
    world_vy: float = 0.0


class DrivePreview50:
    """Source-driven editor kinematics.

    Ordinary travel is chassis-aligned. Powerslide is a separate state.  v0.51
    also mirrors the recovered motorcycle animation inputs:
      speedRatio = clamp(speedKPH / 50, 0, 1)
      leanInput  = speedRatio * steeringInput
      powerslide animation blend changes at 4 units/second.
    """
    POWER_SLIDE_MIN_SPEED = 22.222223  # recovered threshold: 80 km/h
    REFERENCE_MAX_STEER_DEG = 30.0     # constructor default in reference build

    def __init__(self, tuning: Dict[str,Any], engine: Dict[str,Any], suspension: Dict[str,Any],
                 wheelbase: float, track_width: float, front_diameter: float, rear_diameter: float,
                 is_motorcycle: bool=False):
        self.tuning=dict(tuning or {})
        self.engine=dict(engine or {})
        self.suspension=dict(suspension or {})
        self.wheelbase=max(0.05,float(wheelbase or 0.05))
        self.track_width=max(0.0,float(track_width or 0.0))
        self.front_radius=max(0.05,float(front_diameter or 0.8)*0.5)
        self.rear_radius=max(0.05,float(rear_diameter or front_diameter or 0.8)*0.5)
        self.is_motorcycle=bool(is_motorcycle)
        self.state=DriveState()
        # Editor-only feel controls. They never write back to the package.
        self.preview_response_scale=1.0

    @staticmethod
    def _f(d,k,default):
        try: return float(d.get(k,default))
        except Exception: return float(default)

    @property
    def max_forward_mps(self):
        return max(1.0,self._f(self.tuning,"Speed",100.0)/3.6)

    @property
    def max_reverse_mps(self):
        return max(1.0,self._f(self.engine,"Max Reverse Speed KPH",30.0)/3.6)

    @property
    def max_steer_deg(self):
        for key in ("Max Steering Angle","Max Steering Angle Degrees"):
            if key in self.suspension:
                return max(1.0,abs(self._f(self.suspension,key,self.REFERENCE_MAX_STEER_DEG)))
        return self.REFERENCE_MAX_STEER_DEG

    def reset(self):
        self.state=DriveState()

    @staticmethod
    def _native_planar_angle(normal, forward, velocity) -> float:
        """Match VuVehicleUtil::calcPowerSlideAngle's signed plane angle.

        The native routine projects both vectors onto the plane orthogonal to
        ``normal``, normalizes them, uses acos(abs-safe dot), then applies the
        sign from the oriented cross product.  This is used for the preview's
        reported/driver powerslide angle instead of fabricating one from a
        steering coefficient.
        """
        nx,ny,nz=[float(x) for x in normal]
        fx,fy,fz=[float(x) for x in forward]
        vx,vy,vz=[float(x) for x in velocity]
        ndotf=nx*fx+ny*fy+nz*fz
        ndotv=nx*vx+ny*vy+nz*vz
        af=(fx-nx*ndotf, fy-ny*ndotf, fz-nz*ndotf)
        av=(vx-nx*ndotv, vy-ny*ndotv, vz-nz*ndotv)
        lf=math.sqrt(sum(c*c for c in af)); lv=math.sqrt(sum(c*c for c in av))
        if lf<=1e-6 or lv<=1e-6:
            return 0.0
        af=tuple(c/lf for c in af); av=tuple(c/lv for c in av)
        dot=max(-1.0,min(1.0,sum(a*b for a,b in zip(af,av))))
        angle=math.acos(dot)
        cx=af[1]*av[2]-af[2]*av[1]
        cy=af[2]*av[0]-af[0]*av[2]
        cz=af[0]*av[1]-af[1]*av[0]
        orient=cx*nx+cy*ny+cz*nz
        return angle if orient>=0.0 else -angle

    def _ackermann(self, center_deg: float) -> Tuple[float,float]:
        a=math.radians(center_deg)
        if abs(a)<1e-6 or self.track_width<=1e-6:
            return center_deg,center_deg
        R=self.wheelbase/max(1e-6,abs(math.tan(a)))
        half=self.track_width*0.5
        inner=math.degrees(math.atan(self.wheelbase/max(1e-4,R-half)))
        outer=math.degrees(math.atan(self.wheelbase/(R+half)))
        if center_deg>0: return inner,outer
        return -outer,-inner

    def step(self,dt: float, throttle: float, brake_reverse: float, steer: float, powerslide: bool, boost: bool=False):
        """Advance the authored-tuning dynamic preview.

        This is intentionally no longer the v0.50-v0.59 kinematic "set a slip
        angle" approximation.  The Studio does not contain Vector Unit's native
        PhysX scene, so it cannot literally call VuVehicle::onApplyForces.  It
        *can*, however, use the tuning assigned to each vehicle to drive a real
        lateral-force bicycle model: mass, traction, steering/lag, drag and the
        induced/power-slide coefficients all affect the result.
        """
        dt=max(0.0,min(float(dt),0.05)); s=self.state
        throttle=max(0.0,min(1.0,float(throttle)))
        brake_reverse=max(0.0,min(1.0,float(brake_reverse)))
        steer=max(-1.0,min(1.0,float(steer)))
        s.steering_input_raw=steer

        # ------------------------------ authored steering + high-speed authority
        # Keyboard A/D is a step input whereas the game normally receives an
        # analog steering axis. Map the tiny authored Steering Lag value to a
        # deliberate rack response, then let tire forces (not an arbitrary yaw
        # formula) decide how quickly the chassis actually rotates.
        authored_lag=max(0.001,self._f(self.tuning,"Steering Lag",0.10))
        response=max(0.5,min(2.0,float(getattr(self,"preview_response_scale",1.0))))
        # v0.72 used 0.55 + lag*3 seconds, which made keyboard steering visibly
        # trail the front wheels/driver pose. Keep authored lag as a modifier,
        # but use a much faster digital-input rack and an even quicker return to
        # centre. This affects the editor preview only.
        if abs(steer)>1e-5:
            steer_tau=(0.115 + authored_lag*0.55)/response
        else:
            steer_tau=(0.075 + authored_lag*0.30)/response
        alpha=1.0-math.exp(-dt/max(0.02,steer_tau))
        s.steering_input += (steer-s.steering_input)*alpha
        if abs(steer)<1e-5 and abs(s.steering_input)<0.001:
            s.steering_input=0.0

        steer_strength=max(0.05,self._f(self.tuning,"Steering",100.0))/100.0
        speed_ratio=min(1.25,abs(s.speed_mps)/max(1.0,self.max_forward_mps))
        # Preserve speed-sensitive steering but retain enough authority for an
        # editor keyboard at race speed. Tire forces still limit the real yaw.
        authority=max(0.30,1.0/(1.0+1.85*(speed_ratio**1.55)))
        requested=s.steering_input*self.max_steer_deg*steer_strength*authority
        steer_angle_tau=max(0.035,steer_tau*0.32)
        s.steering_deg += (requested-s.steering_deg)*(1.0-math.exp(-dt/steer_angle_tau))
        s.front_left_steer_deg,s.front_right_steer_deg=self._ackermann(s.steering_deg)

        # ------------------------------------------- longitudinal assigned tuning
        vmax=self.max_forward_mps*(1.18 if boost else 1.0)
        vrev=self.max_reverse_mps
        accel_factor=max(0.05,self._f(self.tuning,"Accel Factor",1.0))
        accel_stat=max(0.0,self._f(self.tuning,"Acceleration",70.0))/100.0
        drag=max(0.0,self._f(self.tuning,"Drag Coeff",0.2))
        # Slightly stronger preview acceleration keeps keyboard testing from
        # feeling like slow motion while still respecting per-vehicle tuning.
        max_drive_accel=(4.8 + 6.4*accel_factor*(0.45+0.55*accel_stat))*max(0.75,min(1.6,float(getattr(self,"preview_response_scale",1.0))))

        braking=bool(brake_reverse>0.0 and s.speed_mps>0.35)
        s.braking=braking
        brake_target=1.0 if braking else 0.0
        s.brake_intensity += (brake_target-s.brake_intensity)*(1.0-math.exp(-dt/0.09))

        u=float(s.speed_mps)
        if throttle>0.0:
            # Engine acceleration tapers smoothly into the authored top speed.
            ratio=max(0.0,u/max(1.0,vmax))
            ax=max_drive_accel*throttle*max(0.0,1.0-ratio**1.45)
            if u < 0.0:
                ax += 10.0*throttle
        elif braking:
            ax=-min(15.0,9.5+3.0*accel_factor)*brake_reverse
        elif brake_reverse>0.0:
            # Reverse has its own authored speed cap.
            rratio=max(0.0,(-u)/max(1.0,vrev))
            ax=-(5.0+2.0*accel_factor)*brake_reverse*max(0.0,1.0-rratio**1.3)
        else:
            # Rolling + aerodynamic loss.
            ax=-(0.22+drag*0.10*u*abs(u)/max(1.0,self.max_forward_mps))* (1.0 if u>0 else -1.0 if u<0 else 0.0)

        # Native vehicle data includes a power boost while sliding. Apply it as
        # longitudinal force, not as a fabricated extra slip angle.
        if s.powersliding and throttle>0.0:
            ps_boost=max(1.0,self._f(self.engine,"Power Slide Power Boost",1.0))
            ax += max_drive_accel*0.22*(ps_boost-1.0)*throttle

        u += ax*dt
        if braking and u<0.0: u=0.0
        u=max(-vrev,min(vmax,u))
        if abs(u)<0.015 and throttle<=0.0 and brake_reverse<=0.0: u=0.0
        s.speed_mps=u

        # ----------------------------------------- power-slide state from runtime
        # 80 km/h is recovered from VuVehicle. Use the *smoothed* steering input
        # so pressing Space does not instantly snap the car into a slide.
        can_slide=bool(
            powerslide and u>=self.POWER_SLIDE_MIN_SPEED and
            abs(s.steering_input)>=0.22
        )
        s.powersliding=can_slide
        s.powerslide_anim_blend=(
            min(1.0,s.powerslide_anim_blend+dt*4.0) if can_slide
            else max(0.0,s.powerslide_anim_blend-dt*4.0)
        )

        # ---------------------------------------------- lateral tire-force model
        m=max(100.0,self._f(self.tuning,"Mass",1000.0))
        traction=max(0.15,self._f(self.tuning,"Traction",1.0))
        L=max(0.35,self.wheelbase)
        a=L*0.50; b=L-a
        track=max(0.5,self.track_width if self.track_width>0.0 else L*0.55)
        Iz=max(100.0,m*0.34*(L*L+track*track))
        v=float(s.lateral_mps)        # +left
        r=float(s.yaw_rate_rps)      # +left yaw

        if abs(u)<2.2:
            # Tire slip formulas are singular near zero speed. Blend to a
            # gentle low-speed kinematic response and erase residual drift.
            low=max(0.0,min(1.0,abs(u)/2.2))
            kin=(u/L)*math.tan(math.radians(s.steering_deg))
            r += (kin-r)*(1.0-math.exp(-dt*5.0*max(0.15,low)))
            v *= math.exp(-dt*5.5)
        elif u>0.0:
            delta=math.radians(s.steering_deg)
            safe_u=max(2.0,abs(u))
            alpha_f=delta-math.atan2(v+a*r,safe_u)
            alpha_r=-math.atan2(v-b*r,safe_u)

            # Build cornering stiffness from the assigned traction and vehicle
            # weight, then saturate each axle at its friction limit. This is the
            # key difference from assigning slip_deg by hand.
            g=9.81
            Fzf=m*g*(b/L); Fzr=m*g*(a/L)
            mu_front=traction
            induced_grip=max(0.05,min(1.0,self._f(self.tuning,"Induced Power Slide Traction Factor",0.25)))
            mu_rear=traction*(induced_grip if can_slide else 1.0)
            Cf=(Fzf*mu_front)/math.radians(7.5)
            Cr=(Fzr*max(0.15,traction))/math.radians(8.5)
            Fyf=max(-Fzf*mu_front,min(Fzf*mu_front,Cf*alpha_f))
            Fyr=max(-Fzr*mu_rear,min(Fzr*mu_rear,Cr*alpha_r))

            dv=(Fyf+Fyr)/m-u*r
            dr=(a*Fyf-b*Fyr)/Iz

            if can_slide:
                induced=max(0.0,self._f(self.tuning,"Induced Power Slide Coeff",1.0))
                ps=max(0.0,self._f(self.tuning,"Power Slide Coeff",0.5))
                if boost:
                    ps=max(ps,self._f(self.tuning,"Buffed Power Slide Coeff",ps))
                sr=min(1.2,u/max(1.0,self.max_forward_mps))
                # The induced coefficients create a yaw moment/lateral load
                # transfer while rear traction is reduced above. Coefficients
                # never become a literal angle.
                dr += s.steering_input*induced*(0.85+0.55*ps)*sr
                dv += s.steering_input*induced*(0.35+0.30*ps)*sr
            else:
                # Assigned traction recentres the chassis after a slide.
                dv -= v*(0.55+0.45*traction)
                # With the steering released the native tire model rapidly
                # self-aligns rather than carrying a permanent sideways drift.
                # Apply that recovery only near centre so ordinary cornering
                # remains governed by the axle forces above.
                if abs(s.steering_input) < 0.10:
                    dv -= v*(1.35+0.85*traction)
                    dr -= r*(1.10+0.65*traction)

            v += dv*dt
            r += dr*dt
            # A friction-limited vehicle cannot sustain yaw acceleration whose
            # required lateral acceleration exceeds mu*g.  Enforcing that
            # physical bound is what the old bicycle-yaw shortcut was missing
            # at 150-200 km/h. Allow a little extra during an induced slide.
            yaw_mu=traction*(1.18 if can_slide else 1.02)
            yaw_limit=max(0.10,yaw_mu*g/max(3.0,abs(u)))
            r=max(-yaw_limit,min(yaw_limit,r))
            # Velocity bound is only a numerical guard; actual slip comes from
            # the integrated tire-force imbalance above.
            v=max(-abs(u)*0.70,min(abs(u)*0.70,v))
        else:
            # Reverse preview: keep the car controllable but do not fabricate a
            # backwards powerslide model.
            kin=(u/L)*math.tan(math.radians(s.steering_deg))
            r += (kin-r)*(1.0-math.exp(-dt*2.5))
            v *= math.exp(-dt*3.5)

        s.lateral_mps=v; s.yaw_rate_rps=r
        if abs(u)>0.05:
            # Local +Y is vehicle-forward and +X is vehicle-right. ``v`` is
            # positive to the left, hence local velocity X is -v.
            s.slip_deg=math.degrees(self._native_planar_angle(
                (0.0,0.0,1.0),(0.0,1.0,0.0),(-v,u,0.0)
            ))
        else:
            s.slip_deg=0.0
        if abs(s.slip_deg)<0.02: s.slip_deg=0.0
        s.yaw_deg=(s.yaw_deg+math.degrees(r*dt))%360.0

        # Body local +Y is forward and +X is right. Lateral state above is
        # +left, so local velocity is [-v, u]. Rotate it by the same chassis
        # matrix used for rendering; this removes the old slide/render mismatch.
        ya=math.radians(s.yaw_deg); cy=math.cos(ya); sy=math.sin(ya)
        local_right=-v; local_forward=u
        wx=cy*local_right-sy*local_forward
        wy=sy*local_right+cy*local_forward
        s.world_vx=wx; s.world_vy=wy
        s.x += wx*dt; s.y += wy*dt

        dist=u*dt
        s.wheel_spin_front_deg=(s.wheel_spin_front_deg+math.degrees(dist/self.front_radius))%360.0
        s.wheel_spin_rear_deg=(s.wheel_spin_rear_deg+math.degrees(dist/self.rear_radius))%360.0

        # ------------------------------------------ recovered driver animation IO
        longitudinal_turn=max(-1.0,min(1.0,u))
        if self.is_motorcycle:
            s.motorcycle_speed_ratio=max(0.0,min(1.0,u*3.6/50.0))
            lean_target=s.motorcycle_speed_ratio*s.steering_input
            omega=20.0; x=dt*omega
            inv=1.0/(1.0+x+x*x*0.48+x*x*x*0.235)
            error=s.motorcycle_lean_input-lean_target
            temp=dt*(s.motorcycle_lean_velocity+error*omega)
            s.motorcycle_lean_input=lean_target+(error+temp)*inv
            s.motorcycle_lean_velocity=(s.motorcycle_lean_velocity-temp*omega)*inv
        else:
            s.motorcycle_speed_ratio=0.0
            s.motorcycle_lean_input=0.0
            s.motorcycle_lean_velocity=0.0

        normal_turn=s.steering_input*longitudinal_turn*(1.0-s.motorcycle_speed_ratio)
        max_angle=max(1e-4,self.max_steer_deg)
        slide_angle=max(-max_angle,min(max_angle,s.slip_deg))
        native_slide_turn=-slide_angle/max_angle
        # During a powerslide the recovered slip-angle control can briefly have
        # the opposite sign while the driver is still holding a turn. Blending
        # that value directly made the hands snap across the wheel. Keep the
        # slide magnitude but preserve the held steering side until the input is
        # nearly centred; then let the native slip sign take over.
        if abs(s.steering_input)>0.08 and abs(native_slide_turn)>0.02:
            slide_turn=math.copysign(abs(native_slide_turn),s.steering_input)
        else:
            slide_turn=native_slide_turn
        blend_anim=max(0.0,min(1.0,s.powerslide_anim_blend))
        target_turn=max(-1.0,min(1.0,normal_turn*(1.0-blend_anim)+slide_turn*blend_anim))
        # A short visual filter removes frame-to-frame hand jitter without
        # reintroducing the long v0.72 steering delay.
        turn_alpha=1.0-math.exp(-dt/0.055)
        s.driver_turn_input += (target_turn-s.driver_turn_input)*turn_alpha
        s.driver_turn_input=max(-1.0,min(1.0,s.driver_turn_input))
        return s
