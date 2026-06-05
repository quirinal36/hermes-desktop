import { Suspense, useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Environment, Lightformer } from "@react-three/drei";
import { configureTextBuilder } from "troika-three-text";
import * as THREE from "three";
import { AgentModel } from "./objects/agents";
import { RIGGED_EMPLOYEE_URL, RIGGED_MAN_URL } from "./objects/RiggedCharacter";
import { Workstations, FurniturePieces } from "./objects/furniture";
import {
  buildWorkstations,
  REST_SEATS,
  REST_FURNITURE,
  EXECUTIVE_DECOR,
  INTERIOR_WALLS,
  DIVIDER_X,
  DOOR_Y,
  type Workstation,
  type Seat,
} from "./layout";
import { WORLD_W, WORLD_H, WALK_SPEED, SCALE } from "./core/constants";
import { toWorld } from "./core/geometry";
import type { OfficeAgent, RenderAgent } from "./core/types";
import officeFontUrl from "../../../assets/fonts/Manrope-Medium.ttf";
import { useTheme } from "../../../components/ThemeProvider";
import { THEMES } from "../../../constants";

// drei's <Text> (agent nameplates / speech bubbles, via troika) defaults to two
// behaviours the renderer's strict CSP (`script-src`/`default-src 'self'`)
// blocks: spawning a blob-backed Web Worker, and fetching its default font from
// a CDN. Disable the worker (typeset on the main thread) and point troika at
// our locally-bundled Manrope so labels render fully offline without loosening
// the app's Content-Security-Policy.
configureTextBuilder({ useWorker: false, defaultFontURL: officeFontUrl });

// Walking speed (canvas units / second) and arrival threshold.
const WALK_UNITS_PER_SEC = 130;
const ARRIVE_DISTANCE = 8;

// The world's day/night look (floor, walls, lighting) is driven by the system
// clock, NOT the app's UI theme — so future 3D worlds can reuse this same
// time-of-day model. Only the canvas background follows the app theme.
interface WorldPalette {
  floor: string;
  rug: string;
  wallNS: string;
  wallEW: string;
  hemiSky: string;
  hemiGround: string;
  hemiIntensity: number;
  ambient: number;
  directional: number;
  // Image-based-lighting (Lightformer environment) strength + warmth. With
  // ACES tone mapping the punchier directional + soft IBL replace the old flat
  // fill, so ambient/hemi are dialled down to avoid washing the scene out.
  envIntensity: number;
  keyColor: string;
}

const DAY_PALETTE: WorldPalette = {
  floor: "#e7e2d8",
  rug: "#cdd7e5",
  wallNS: "#c9c2b4",
  wallEW: "#d2ccbf",
  hemiSky: "#ffffff",
  hemiGround: "#b9b4a8",
  hemiIntensity: 0.45,
  ambient: 0.22,
  directional: 2.0,
  envIntensity: 0.75,
  keyColor: "#fff4e2",
};

const NIGHT_PALETTE: WorldPalette = {
  floor: "#262a31",
  rug: "#313845",
  wallNS: "#2f333b",
  wallEW: "#363b44",
  hemiSky: "#3a4150",
  hemiGround: "#101216",
  hemiIntensity: 0.3,
  ambient: 0.14,
  directional: 1.1,
  envIntensity: 0.32,
  keyColor: "#cdd6ff",
};

// Only the canvas background follows the app's light/dark theme.
const THEME_BACKGROUND = { light: "#f3f1ec", dark: "#16181d" } as const;

// Daytime: 06:00–17:59 local. Drives the day/night world palette.
function isDaytime(date: Date = new Date()): boolean {
  const hour = date.getHours();
  return hour >= 6 && hour < 18;
}

type ControllerMode = "toSeat" | "seated";
interface ControllerState {
  mode: ControllerMode;
  /** Which seat the agent is currently heading to / sitting at. */
  goalKey: "desk" | "rest" | null;
}

function randomBetween(min: number, max: number): number {
  return min + Math.random() * (max - min);
}

// Doorway waypoints just inside each room, so agents pass through the gap in
// the partition instead of clipping the wall (we have no full pathfinder).
function routeTarget(
  ax: number,
  finalX: number,
  finalY: number,
): { x: number; y: number } {
  const onEast = ax > DIVIDER_X;
  const targetEast = finalX > DIVIDER_X;
  if (onEast === targetEast) return { x: finalX, y: finalY };
  return { x: targetEast ? DIVIDER_X + 60 : DIVIDER_X - 60, y: DOOR_Y };
}

function makeRenderAgent(agent: OfficeAgent): RenderAgent {
  // Spawn near the entrance (south edge); the controller routes the agent to
  // its assigned desk from there.
  const x = randomBetween(820, 1000);
  const y = 1650;
  return {
    ...agent,
    x,
    y,
    targetX: x,
    targetY: y,
    path: [],
    facing: Math.PI,
    frame: Math.floor(randomBetween(0, 240)),
    walkSpeed: WALK_SPEED,
    phaseOffset: randomBetween(0, Math.PI * 2),
    state: "standing",
  };
}

/**
 * Holds the live agent simulation. Each agent walks to its desk (gateway up)
 * or to a rest-room beanbag (gateway off) and sits. Positions are mutated
 * in-place on the refs each frame so avatars animate without React re-renders.
 */
function AgentsLayer({
  agents,
  workstations,
  selectedId,
  onSelect,
}: {
  agents: OfficeAgent[];
  workstations: Workstation[];
  selectedId: string | null;
  onSelect: (id: string) => void;
}): React.JSX.Element {
  const agentsRef = useRef<RenderAgent[]>([]);
  const lookupRef = useRef<Map<string, RenderAgent>>(new Map());
  const controllerRef = useRef<Map<string, ControllerState>>(new Map());

  const deskSeatByAgent = useMemo(() => {
    const map = new Map<string, Seat>();
    for (const w of workstations) {
      map.set(w.agentId, { x: w.seatX, y: w.seatY, facing: w.seatFacing });
    }
    return map;
  }, [workstations]);

  // Assign each agent a rest-room beanbag (round-robin) for when its gateway
  // is off.
  const restSeatByAgent = useMemo(() => {
    const map = new Map<string, Seat>();
    if (REST_SEATS.length > 0) {
      agents.forEach((agent, index) => {
        map.set(agent.id, REST_SEATS[index % REST_SEATS.length]);
      });
    }
    return map;
  }, [agents]);

  // Reconcile the simulation list whenever the set of agents changes, keeping
  // existing agents' positions so they don't teleport on a profile refresh.
  useMemo(() => {
    const prev = lookupRef.current;
    const next: RenderAgent[] = agents.map((agent) => {
      const existing = prev.get(agent.id);
      if (existing) {
        return { ...existing, ...agent };
      }
      return makeRenderAgent(agent);
    });
    agentsRef.current = next;
    const lookup = new Map<string, RenderAgent>();
    for (const a of next) lookup.set(a.id, a);
    lookupRef.current = lookup;
    // Drop controller state for removed agents.
    const controller = controllerRef.current;
    for (const id of [...controller.keys()]) {
      if (!lookup.has(id)) controller.delete(id);
    }
  }, [agents]);

  useFrame((_, delta) => {
    const step = Math.min(delta, 0.05); // clamp big frame gaps
    for (const agent of agentsRef.current) {
      agent.frame += step * 60;

      // Working agents (gateway up) sit at their desk; everyone else rests in
      // the rest room.
      const working = agent.status === "working";
      const goalKey: "desk" | "rest" = working ? "desk" : "rest";
      const goal = working
        ? deskSeatByAgent.get(agent.id)
        : restSeatByAgent.get(agent.id);

      let ctrl = controllerRef.current.get(agent.id);
      if (!ctrl) {
        ctrl = { mode: "toSeat", goalKey: null };
        controllerRef.current.set(agent.id, ctrl);
      }

      if (!goal) {
        agent.state = "standing";
        continue;
      }

      // Gateway flipped (profile started/stopped) — head to the new seat.
      if (ctrl.goalKey !== goalKey) {
        ctrl.goalKey = goalKey;
        ctrl.mode = "toSeat";
      }

      const moveToward = (tx: number, ty: number): boolean => {
        const dx = tx - agent.x;
        const dy = ty - agent.y;
        const dist = Math.hypot(dx, dy);
        if (dist <= ARRIVE_DISTANCE) {
          agent.x = tx;
          agent.y = ty;
          return true;
        }
        const move = Math.min(dist, WALK_UNITS_PER_SEC * step);
        agent.x += (dx / dist) * move;
        agent.y += (dy / dist) * move;
        agent.facing = Math.atan2(dx, dy);
        agent.state = "walking";
        return false;
      };

      if (ctrl.mode === "seated") {
        agent.x = goal.x;
        agent.y = goal.y;
        agent.facing = goal.facing;
        agent.state = "sitting";
        continue;
      }

      // Heading to the seat, routing through the doorway when changing rooms.
      const wp = routeTarget(agent.x, goal.x, goal.y);
      const reachedFinal = wp.x === goal.x && wp.y === goal.y;
      if (moveToward(wp.x, wp.y) && reachedFinal) {
        agent.facing = goal.facing;
        agent.state = "sitting";
        ctrl.mode = "seated";
      }
    }
  });

  return (
    <>
      {agents.map((agent) => (
        <AgentModel
          key={agent.id}
          agentId={agent.id}
          name={agent.name}
          // Nameplate shows the name only; the model/provider stays in the
          // selection panel rather than cluttering the 3D head label.
          subtitle={null}
          status={agent.status}
          color={agent.color}
          appearance={agent.avatarProfile}
          agentsRef={agentsRef}
          agentLookupRef={lookupRef}
          onClick={onSelect}
          showSpeech={selectedId === agent.id}
          speechText={selectedId === agent.id ? `Hi, I'm ${agent.name}` : null}
          riggedModelUrl={
            agent.position === "ceo" ? RIGGED_EMPLOYEE_URL : RIGGED_MAN_URL
          }
          riggedModelTint={agent.position === "ceo" ? null : agent.color}
        />
      ))}
    </>
  );
}

/** Floor, rug and perimeter walls — a clean, minimal office shell. */
function Room({ palette }: { palette: WorldPalette }): React.JSX.Element {
  const halfW = WORLD_W / 2;
  const halfH = WORLD_H / 2;
  const wallH = 2.4;
  const wallT = 0.2;
  return (
    <group>
      {/* Floor — slightly glossy so the IBL adds a soft sheen + grounding. */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0, 0]} receiveShadow>
        <planeGeometry args={[WORLD_W, WORLD_H]} />
        <meshStandardMaterial
          color={palette.floor}
          roughness={0.78}
          metalness={0}
          envMapIntensity={0.6}
        />
      </mesh>
      {/* Center rug for a bit of warmth (matte). */}
      <mesh
        rotation={[-Math.PI / 2, 0, 0]}
        position={[0, 0.01, 0]}
        receiveShadow
      >
        <planeGeometry args={[WORLD_W * 0.42, WORLD_H * 0.42]} />
        <meshStandardMaterial
          color={palette.rug}
          roughness={0.95}
          metalness={0}
          envMapIntensity={0.4}
        />
      </mesh>
      {/* Walls */}
      <mesh position={[0, wallH / 2, -halfH]}>
        <boxGeometry args={[WORLD_W, wallH, wallT]} />
        <meshStandardMaterial color={palette.wallNS} />
      </mesh>
      <mesh position={[0, wallH / 2, halfH]}>
        <boxGeometry args={[WORLD_W, wallH, wallT]} />
        <meshStandardMaterial color={palette.wallNS} />
      </mesh>
      <mesh position={[-halfW, wallH / 2, 0]}>
        <boxGeometry args={[wallT, wallH, WORLD_H]} />
        <meshStandardMaterial color={palette.wallEW} />
      </mesh>
      <mesh position={[halfW, wallH / 2, 0]}>
        <boxGeometry args={[wallT, wallH, WORLD_H]} />
        <meshStandardMaterial color={palette.wallEW} />
      </mesh>
    </group>
  );
}

/** Interior partition walls (e.g. the work-area / rest-room divider). */
function InteriorWalls({
  palette,
}: {
  palette: WorldPalette;
}): React.JSX.Element {
  const wallH = 2.4;
  return (
    <group>
      {INTERIOR_WALLS.map((wall) => {
        const [cx, , cz] = toWorld(wall.x + wall.w / 2, wall.y + wall.h / 2);
        return (
          <mesh key={wall.id} position={[cx, wallH / 2, cz]} castShadow>
            <boxGeometry args={[wall.w * SCALE, wallH, wall.h * SCALE]} />
            <meshStandardMaterial color={palette.wallEW} />
          </mesh>
        );
      })}
    </group>
  );
}

/**
 * The native, in-renderer 3D office. Replaces the old webview that pointed at a
 * separately-cloned hermes-office dev server. Each agent corresponds to a
 * desktop profile.
 */
export default function Office3D({
  agents,
  selectedId,
  onSelectAgent,
}: {
  agents: OfficeAgent[];
  selectedId: string | null;
  onSelectAgent: (id: string | null) => void;
}): React.JSX.Element {
  // Clicking the selected agent again clears the selection.
  const handleSelect = (id: string): void => {
    onSelectAgent(id === selectedId ? null : id);
  };

  // The CEO (if any) gets a separate executive desk; everyone else grids up.
  const ceoId = useMemo(
    () => agents.find((a) => a.position === "ceo")?.id ?? null,
    [agents],
  );

  // One desk per agent, assigned in profile order.
  const workstations = useMemo(
    () =>
      buildWorkstations(
        agents.map((a) => a.id),
        ceoId,
      ),
    [agents, ceoId],
  );

  // Only the background follows the app's light/dark theme.
  const { resolved } = useTheme();
  const background = useMemo(() => {
    const def = THEMES.find((th) => th.id === resolved);
    return def?.appearance === "light"
      ? THEME_BACKGROUND.light
      : THEME_BACKGROUND.dark;
  }, [resolved]);

  // The world's day/night look follows the system clock, re-checked each
  // minute so the scene flips at dawn/dusk without a manual refresh.
  const [daytime, setDaytime] = useState(() => isDaytime());
  useEffect(() => {
    const id = setInterval(() => setDaytime(isDaytime()), 60_000);
    return () => clearInterval(id);
  }, []);
  const palette = daytime ? DAY_PALETTE : NIGHT_PALETTE;

  return (
    <Canvas
      shadows="percentage"
      dpr={[1, 2]}
      camera={{ position: [0, 22, 26], fov: 50 }}
      gl={{
        antialias: true,
        toneMapping: THREE.ACESFilmicToneMapping,
        toneMappingExposure: 1.05,
      }}
      onPointerMissed={() => onSelectAgent(null)}
      style={{ width: "100%", height: "100%" }}
    >
      <color attach="background" args={[background]} />
      {/* Soft image-based lighting baked once from in-scene Lightformers — no
          external HDRI fetch, so it stays within the renderer's strict CSP. */}
      <Environment frames={1} resolution={256} background={false}>
        <Lightformer
          form="rect"
          intensity={palette.envIntensity}
          color={palette.keyColor}
          position={[0, 20, 0]}
          rotation={[Math.PI / 2, 0, 0]}
          scale={[36, 36, 1]}
        />
        <Lightformer
          form="rect"
          intensity={palette.envIntensity * 0.6}
          color="#eaf0ff"
          position={[0, 8, 24]}
          rotation={[0, 0, 0]}
          scale={[36, 14, 1]}
        />
        <Lightformer
          form="rect"
          intensity={palette.envIntensity * 0.4}
          color="#ffffff"
          position={[-24, 9, 0]}
          rotation={[0, Math.PI / 2, 0]}
          scale={[36, 14, 1]}
        />
        <Lightformer
          form="rect"
          intensity={palette.envIntensity * 0.4}
          color="#ffffff"
          position={[24, 9, 0]}
          rotation={[0, -Math.PI / 2, 0]}
          scale={[36, 14, 1]}
        />
      </Environment>
      <hemisphereLight
        args={[palette.hemiSky, palette.hemiGround, palette.hemiIntensity]}
      />
      <ambientLight intensity={palette.ambient} />
      {/* Key light. The shadow camera is sized to the whole room (~32 world
          units across) — the default ±5 frustum only covered the centre, so
          most furniture cast no shadow before. */}
      <directionalLight
        position={[14, 26, 16]}
        intensity={palette.directional}
        color={palette.keyColor}
        castShadow
        shadow-mapSize-width={2048}
        shadow-mapSize-height={2048}
        shadow-bias={-0.0004}
        shadow-normalBias={0.02}
        shadow-camera-near={1}
        shadow-camera-far={90}
        shadow-camera-left={-20}
        shadow-camera-right={20}
        shadow-camera-top={20}
        shadow-camera-bottom={-20}
      />
      <Room palette={palette} />
      <InteriorWalls palette={palette} />
      <Suspense fallback={null}>
        <Workstations workstations={workstations} />
        <FurniturePieces pieces={REST_FURNITURE} />
        {ceoId && <FurniturePieces pieces={EXECUTIVE_DECOR} />}
      </Suspense>
      <AgentsLayer
        agents={agents}
        workstations={workstations}
        selectedId={selectedId}
        onSelect={handleSelect}
      />
      <OrbitControls
        makeDefault
        enablePan
        minDistance={8}
        maxDistance={48}
        maxPolarAngle={Math.PI / 2.15}
        target={new THREE.Vector3(0, 0, 0)}
      />
    </Canvas>
  );
}
