import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const SIZE = 20;
const CELL_METERS = 10;
const MOVES = [[-1, 0], [1, 0], [0, -1], [0, 1]];
const PROFILES = {
  mars: {
    name: "MARTE", mode: "TERRENO DE EXPLORACIÓN", surface: 0x80584a,
    background: 0x000000, atmosphere: 0xc37958, roughness: 0.36,
    notice: "Perfil visual de Marte. Parámetros de terreno y riesgo: supuestos de simulación.",
  },
  mercury: {
    name: "MERCURIO", mode: "ROVER HIPOTÉTICO", surface: 0x686b67,
    background: 0x000000, atmosphere: 0xa2a39b, roughness: 0.48,
    notice: "Perfil hipotético: el contraste térmico no se modela en esta versión.",
  },
  earth: {
    name: "TIERRA", mode: "TERRENO DE EXPLORACIÓN", surface: 0x5c7552,
    background: 0x000000, atmosphere: 0x76adbd, roughness: 0.2,
    notice: "Perfil visual de Tierra. Los riesgos son parámetros pedagógicos.",
  },
  venus: {
    name: "VENUS", mode: "ROVER HIPOTÉTICO", surface: 0x987143,
    background: 0x000000, atmosphere: 0xd49b56, roughness: 0.24,
    notice: "Perfil de misión hipotética de corta duración; no representa un rover real.",
  },
  jupiter: {
    name: "JÚPITER", mode: "SONDA ATMOSFÉRICA · MAPA VIRTUAL", surface: 0x806348,
    background: 0x000000, atmosphere: 0xe7a977, roughness: 0.12, gasGiant: true,
    notice: "Gigante gaseoso: la cuadrícula es abstracta. No existe suelo para rover de ruedas.",
  },
  saturn: {
    name: "SATURNO", mode: "SONDA ATMOSFÉRICA · MAPA VIRTUAL", surface: 0x8c7854,
    background: 0x000000, atmosphere: 0xe1c38a, roughness: 0.12, gasGiant: true,
    notice: "Gigante gaseoso: simulación abstracta de sonda; no se modelan superficie ni anillos físicos.",
  },
  uranus: {
    name: "URANO", mode: "SONDA ATMOSFÉRICA · MAPA VIRTUAL", surface: 0x527a7d,
    background: 0x000000, atmosphere: 0x73c1c5, roughness: 0.1, gasGiant: true,
    notice: "Gigante helado: la cuadrícula es abstracta. No existe suelo para rover de ruedas.",
  },
  neptune: {
    name: "NEPTUNO", mode: "SONDA ATMOSFÉRICA · MAPA VIRTUAL", surface: 0x405987,
    background: 0x000000, atmosphere: 0x638bd9, roughness: 0.14, gasGiant: true,
    notice: "Gigante helado: simulación abstracta de sonda; no se modela suelo sólido.",
  },
};

const el = (id) => document.getElementById(id);
const sceneElement = el("scene");
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 160);
camera.position.set(17, 20, 23);
const SUN_DISTANCES = {
  mercury: 54,
  venus: 61,
  earth: 66,
  mars: 74,
  jupiter: 88,
  saturn: 98,
  uranus: 110,
  neptune: 120,
};
const sunViewingDirection = new THREE.Vector3(-5, 0, -23)
  .sub(camera.position)
  .normalize();
const initialSunPosition = camera.position.clone().addScaledVector(
  sunViewingDirection,
  SUN_DISTANCES.earth,
);

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.13;
sceneElement.appendChild(renderer.domElement);

const orbit = new OrbitControls(camera, renderer.domElement);
orbit.enableDamping = true;
orbit.dampingFactor = 0.08;
orbit.minDistance = 13;
orbit.maxDistance = 45;
orbit.maxPolarAngle = Math.PI * 0.46;
orbit.target.set(0, 0, 0);

scene.add(new THREE.HemisphereLight(0xc9d7d0, 0x282522, 1.35));
const keyLight = new THREE.DirectionalLight(0xffe6c6, 2.6);
keyLight.position.copy(initialSunPosition);
scene.add(keyLight);
const fillLight = new THREE.DirectionalLight(0x91b8bd, 0.45);
fillLight.position.set(10, 8, -8);
scene.add(fillLight);

const mapRoot = new THREE.Group();
scene.add(mapRoot);
const worldRoot = new THREE.Group();
scene.add(worldRoot);
const routeRoot = new THREE.Group();
scene.add(routeRoot);
const beliefRoot = new THREE.Group();
scene.add(beliefRoot);
const cosmosRoot = new THREE.Group();
scene.add(cosmosRoot);
const scienceRoot = new THREE.Group();
scene.add(scienceRoot);

const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
const dragPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
const dragPoint = new THREE.Vector3();
const state = {
  planet: "mars",
  tool: "rock",
  strategy: "safe",
  start: { row: 18, col: 1 },
  goal: { row: 1, col: 18 },
  rover: { row: 18, col: 1 },
  energy: 100,
  steps: 0,
  objects: new Map(),
  objectCounter: 0,
  distanceRoute: null,
  safeRoute: null,
  activeRoute: null,
  playing: false,
  timer: null,
  roverFrom: null,
  roverTo: null,
  roverMoveStarted: 0,
  terrain: null,
  grid: null,
  planetVisual: null,
  dragged: null,
  dragStartPointer: null,
  dragMoved: false,
  toastTimeout: null,
  pythonState: null,
  requestPending: false,
  stepPending: false,
  experimentTimer: null,
  configTimer: null,
  meteorTimer: 0,
  meteors: [],
  starField: null,
  sun: null,
  sunGlow: null,
  moonSystem: null,
};

const materials = {
  rock: new THREE.MeshStandardMaterial({ color: 0x89857e, roughness: 0.97 }),
  rockWarm: new THREE.MeshStandardMaterial({ color: 0x9a7863, roughness: 0.96 }),
  rockCool: new THREE.MeshStandardMaterial({ color: 0x767a78, roughness: 0.98 }),
  pebble: new THREE.MeshStandardMaterial({ color: 0xaaa095, roughness: 0.98 }),
  craterFloor: new THREE.MeshStandardMaterial({ color: 0x29221e, roughness: 1 }),
  craterRim: new THREE.MeshStandardMaterial({ color: 0x88705d, roughness: 0.98 }),
  rough: new THREE.MeshStandardMaterial({ color: 0xd0a25a, roughness: 0.94 }),
  gas: new THREE.MeshStandardMaterial({
    color: 0x85bed4, transparent: true, opacity: 0.2, roughness: 0.26,
    depthWrite: false,
  }),
  gasGround: new THREE.MeshBasicMaterial({
    color: 0x77bad1, transparent: true, opacity: 0.32, depthWrite: false,
    side: THREE.DoubleSide,
    map: createGasHazeTexture(),
  }),
  cliffFloor: new THREE.MeshStandardMaterial({ color: 0x292421, roughness: 1 }),
  cliffWall: new THREE.MeshStandardMaterial({
    color: 0x775847, roughness: 1, flatShading: true, side: THREE.DoubleSide,
  }),
  cliffStrata: new THREE.MeshStandardMaterial({ color: 0x9a7459, roughness: 1 }),
};

class MinHeap {
  constructor() { this.items = []; }

  push(item) {
    this.items.push(item);
    let index = this.items.length - 1;
    while (index > 0) {
      const parent = Math.floor((index - 1) / 2);
      if (this.items[parent].cost <= item.cost) break;
      this.items[index] = this.items[parent];
      index = parent;
    }
    this.items[index] = item;
  }

  pop() {
    if (this.items.length === 0) return null;
    const root = this.items[0];
    const tail = this.items.pop();
    if (this.items.length && tail) {
      let index = 0;
      while (true) {
        const left = index * 2 + 1;
        const right = left + 1;
        if (left >= this.items.length) break;
        const child = right < this.items.length
          && this.items[right].cost < this.items[left].cost ? right : left;
        if (this.items[child].cost >= tail.cost) break;
        this.items[index] = this.items[child];
        index = child;
      }
      this.items[index] = tail;
    }
    return root;
  }

  get length() { return this.items.length; }
}

function cellKey(cell) { return `${cell.row},${cell.col}`; }
function sameCell(a, b) { return a.row === b.row && a.col === b.col; }
function inBounds(row, col) { return row >= 0 && row < SIZE && col >= 0 && col < SIZE; }
function distanceCells(a, b) { return Math.abs(a.row - b.row) + Math.abs(a.col - b.col); }

function smoothNoise(value) { return value * value * (3 - 2 * value); }

function hashNoise(x, z) {
  const value = Math.sin(x * 127.1 + z * 311.7) * 43758.5453;
  return value - Math.floor(value);
}

function valueNoise(x, z) {
  const x0 = Math.floor(x);
  const z0 = Math.floor(z);
  const tx = smoothNoise(x - x0);
  const tz = smoothNoise(z - z0);
  const top = THREE.MathUtils.lerp(
    hashNoise(x0, z0),
    hashNoise(x0 + 1, z0),
    tx,
  );
  const bottom = THREE.MathUtils.lerp(
    hashNoise(x0, z0 + 1),
    hashNoise(x0 + 1, z0 + 1),
    tx,
  );
  return THREE.MathUtils.lerp(top, bottom, tz);
}

function cellPosition(cell) {
  return {
    x: cell.col - SIZE / 2 + 0.5,
    z: cell.row - SIZE / 2 + 0.5,
  };
}

function terrainHeight(x, z) {
  const profile = PROFILES[state.planet];
  const broad = valueNoise(x * 0.18 + 14, z * 0.18 + 8) - 0.5;
  const medium = valueNoise(x * 0.48 + 31, z * 0.48 + 19) - 0.5;
  const detail = valueNoise(x * 1.15 + 5, z * 1.15 + 37) - 0.5;
  const ridges = Math.abs(valueNoise(x * 0.29 + 51, z * 0.29 + 11) - 0.5);
  const canyonCenter = Math.sin(x * 0.34 + 0.5) * 2.8
    + Math.sin(x * 0.78 + 1.2) * 0.72;
  const canyonDistance = Math.abs(z - canyonCenter);
  const canyonBed = Math.exp(-(canyonDistance ** 2) / 0.72);
  const canyonBank = Math.exp(-((canyonDistance - 1.05) ** 2) / 0.2);
  const relief = broad * 1.45
    + medium * 0.82
    + detail * 0.28
    + ridges * 0.45
    + canyonBank * 0.38
    - canyonBed * 0.92;
  return profile.roughness * relief;
}

function createTerrainTexture(baseColor) {
  const size = 256;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const context = canvas.getContext("2d");
  if (!context) throw new Error("No se pudo crear la textura del terreno.");
  const image = context.createImageData(size, size);
  const base = [
    (baseColor >> 16) & 0xff,
    (baseColor >> 8) & 0xff,
    baseColor & 0xff,
  ];
  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      const broad = valueNoise(x / 34 + 13, y / 34 + 7);
      const medium = valueNoise(x / 11 + 29, y / 11 + 17);
      const grain = hashNoise(x + 97, y + 23);
      const shade = 0.68 + broad * 0.28 + medium * 0.18 + grain * 0.08;
      const offset = (y * size + x) * 4;
      for (let channel = 0; channel < 3; channel += 1) {
        image.data[offset + channel] = Math.max(
          0,
          Math.min(255, base[channel] * shade),
        );
      }
      image.data[offset + 3] = 255;
    }
  }
  context.putImageData(image, 0, 0);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.anisotropy = Math.min(8, renderer.capabilities.getMaxAnisotropy());
  return texture;
}

function createTerrainGrid() {
  const points = [];
  const subdivisions = 8;
  for (let line = 0; line <= SIZE; line += 1) {
    const coordinate = line - SIZE / 2;
    for (let segment = 0; segment < SIZE * subdivisions; segment += 1) {
      const start = -SIZE / 2 + segment / subdivisions;
      const end = start + 1 / subdivisions;
      points.push(
        coordinate, terrainHeight(coordinate, start) + 0.035, start,
        coordinate, terrainHeight(coordinate, end) + 0.035, end,
        start, terrainHeight(start, coordinate) + 0.035, coordinate,
        end, terrainHeight(end, coordinate) + 0.035, coordinate,
      );
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(points, 3));
  const material = new THREE.LineBasicMaterial({
    color: 0xb7c7b9,
    transparent: true,
    opacity: 0.18,
    depthWrite: false,
  });
  return new THREE.LineSegments(geometry, material);
}

function createGasHazeTexture() {
  const canvas = document.createElement("canvas");
  canvas.width = 128;
  canvas.height = 128;
  const context = canvas.getContext("2d");
  if (!context) throw new Error("No se pudo crear la textura de la nube de gas.");
  const gradient = context.createRadialGradient(64, 64, 8, 64, 64, 62);
  gradient.addColorStop(0, "rgba(255,255,255,0.82)");
  gradient.addColorStop(0.42, "rgba(255,255,255,0.46)");
  gradient.addColorStop(1, "rgba(255,255,255,0)");
  context.fillStyle = gradient;
  context.fillRect(0, 0, 128, 128);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

function createCraterBasin() {
  const segments = 48;
  const rings = 10;
  const positions = [];
  const indices = [];
  for (let ring = 0; ring <= rings; ring += 1) {
    const radius = ring / rings;
    for (let segment = 0; segment <= segments; segment += 1) {
      const angle = (segment / segments) * Math.PI * 2;
      const uneven = valueNoise(
        Math.cos(angle) * 3 + ring * 0.3 + 41,
        Math.sin(angle) * 3 + ring * 0.3 + 17,
      ) - 0.5;
      const edgeRadius = 0.51 * (1 + uneven * 0.12);
      const normalized = radius;
      const bowl = -0.23 * (1 - smoothNoise(Math.min(1, normalized * 1.4)));
      const rim = 0.095 * Math.exp(-((normalized - 0.88) ** 2) / 0.012);
      positions.push(
        Math.cos(angle) * edgeRadius * radius,
        bowl + rim,
        Math.sin(angle) * edgeRadius * radius,
      );
      if (ring < rings && segment < segments) {
        const current = ring * (segments + 1) + segment;
        const next = current + segments + 1;
        indices.push(current, current + 1, next, next, current + 1, next + 1);
      }
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  return geometry;
}

function createCliffWall(side) {
  const segments = 10;
  const positions = [];
  const indices = [];
  for (let index = 0; index <= segments; index += 1) {
    const z = -0.54 + (index / segments) * 1.08;
    const variation = (hashNoise(index + side * 11, side * 7) - 0.5) * 0.11;
    const topX = side * (0.31 + variation);
    const bottomX = side * (0.43 + variation);
    const topY = 0.12 + hashNoise(index + 19, side + 3) * 0.17;
    const bottomY = -0.38 + hashNoise(index + 37, side + 5) * 0.08;
    positions.push(topX, topY, z, bottomX, bottomY, z);
    if (index < segments) {
      const current = index * 2;
      indices.push(current, current + 1, current + 2, current + 1, current + 3, current + 2);
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  return geometry;
}

function cellWorldPosition(cell, extraHeight = 0) {
  const p = cellPosition(cell);
  return new THREE.Vector3(p.x, terrainHeight(p.x, p.z) + extraHeight, p.z);
}

function createTerrain() {
  if (state.terrain) {
    mapRoot.remove(state.terrain);
    state.terrain.geometry.dispose();
    if (state.terrain.material.map) state.terrain.material.map.dispose();
    state.terrain.material.dispose();
  }
  if (state.grid) {
    mapRoot.remove(state.grid);
    state.grid.geometry.dispose();
    state.grid.material.dispose();
  }

  const profile = PROFILES[state.planet];
  const geometry = new THREE.PlaneGeometry(SIZE, SIZE, 100, 100);
  const positions = geometry.attributes.position;
  for (let i = 0; i < positions.count; i += 1) {
    const x = positions.getX(i);
    const z = -positions.getY(i);
    positions.setZ(i, terrainHeight(x, z));
  }
  geometry.rotateX(-Math.PI / 2);
  geometry.computeVertexNormals();
  const material = new THREE.MeshStandardMaterial({
    map: createTerrainTexture(profile.surface),
    roughness: 0.98,
    metalness: 0,
    side: THREE.DoubleSide,
  });
  state.terrain = new THREE.Mesh(geometry, material);
  state.terrain.receiveShadow = true;
  mapRoot.add(state.terrain);

  state.grid = createTerrainGrid();
  mapRoot.add(state.grid);

  if (state.planetVisual) {
    worldRoot.remove(state.planetVisual);
    state.planetVisual.traverse((child) => {
      if (!child.isMesh) return;
      child.geometry.dispose();
      const childMaterials = Array.isArray(child.material)
        ? child.material
        : [child.material];
      for (const childMaterial of childMaterials) {
        if (childMaterial.map) childMaterial.map.dispose();
        childMaterial.dispose();
      }
    });
  }
  state.planetVisual = new THREE.Mesh(
    new THREE.SphereGeometry(3.1, 48, 32),
    new THREE.MeshStandardMaterial({
      map: createPlanetTexture(
        profile.surface,
        profile.atmosphere,
        Boolean(profile.gasGiant),
      ),
      roughness: 0.96,
      metalness: 0,
    }),
  );
  state.planetVisual.position.set(-14.6, 3.2, -15);
  worldRoot.add(state.planetVisual);
  state.moonSystem = null;
  if (state.planet === "earth") {
    const moonSystem = new THREE.Group();
    const orbitRadius = 4.8;
    const orbitLine = new THREE.Mesh(
      new THREE.TorusGeometry(orbitRadius, 0.012, 4, 96),
      new THREE.MeshBasicMaterial({
        color: 0xc3c8cc,
        transparent: true,
        opacity: 0.34,
        depthWrite: false,
        toneMapped: false,
      }),
    );
    moonSystem.add(orbitLine);
    const moon = new THREE.Mesh(
      new THREE.SphereGeometry(0.78, 32, 24),
      new THREE.MeshStandardMaterial({
        map: createPlanetTexture(0x777a7b, 0xb2b1aa, false),
        roughness: 1,
        metalness: 0,
      }),
    );
    moon.position.set(-orbitRadius, 0, 0);
    moonSystem.add(moon);
    state.planetVisual.add(moonSystem);
    state.moonSystem = moonSystem;
  }
  updateSunForPlanet();

  if (state.planet === "saturn") {
    const ring = new THREE.Mesh(
      new THREE.RingGeometry(3.55, 5.2, 48),
      new THREE.MeshBasicMaterial({
        color: 0xc7ae82, transparent: true, opacity: 0.53, side: THREE.DoubleSide,
      }),
    );
    ring.rotation.x = -Math.PI / 3;
    state.planetVisual.add(ring);
  }
  scene.background = new THREE.Color(profile.background);
  scene.fog = new THREE.Fog(profile.background, 80, 175);

  for (const entity of state.objects.values()) {
    updateEntityPosition(entity);
    if (entity.kind === "gas") {
      entity.group.traverse((child) => {
        if (child.isMesh && child.material === materials.gas) {
          child.material.color.setHex(profile.atmosphere);
        } else if (child.isMesh && child.material === materials.gasGround) {
          child.material.color.setHex(profile.atmosphere);
        }
      });
    }
  }
  if (state.startMarker) updateMarkerPosition(state.startMarker, state.start);
  if (state.goalMarker) updateMarkerPosition(state.goalMarker, state.goal);
  if (state.roverGroup) updateRoverTarget();
}

function createCosmos() {
  const vertices = [];
  for (let index = 0; index < 2400; index += 1) {
    const direction = new THREE.Vector3(
      THREE.MathUtils.randFloatSpread(2),
      THREE.MathUtils.randFloatSpread(2),
      THREE.MathUtils.randFloatSpread(2),
    ).normalize();
    direction.multiplyScalar(THREE.MathUtils.randFloat(88, 98));
    vertices.push(direction.x, direction.y, direction.z);
  }
  const starsGeometry = new THREE.BufferGeometry();
  starsGeometry.setAttribute("position", new THREE.Float32BufferAttribute(vertices, 3));
  const stars = new THREE.Points(
    starsGeometry,
    new THREE.PointsMaterial({
      color: 0xffffff,
      size: 0.22,
      sizeAttenuation: true,
      transparent: true,
      opacity: 1,
      depthWrite: false,
      fog: false,
      toneMapped: false,
    }),
  );
  stars.frustumCulled = false;
  stars.renderOrder = -2;
  cosmosRoot.add(stars);
  state.starField = stars;

  const sun = new THREE.Mesh(
    new THREE.SphereGeometry(1, 48, 32),
    new THREE.MeshBasicMaterial({ color: 0xffdf8c, toneMapped: false }),
  );
  sun.userData.rotationSpeed = 0.012;
  cosmosRoot.add(sun);
  state.sun = sun;
  const sunGlow = new THREE.Sprite(new THREE.SpriteMaterial({
    map: createGlowTexture(),
    color: 0xffc767,
    transparent: true,
    opacity: 0.55,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
    fog: false,
    toneMapped: false,
  }));
  cosmosRoot.add(sunGlow);
  state.sunGlow = sunGlow;
  updateSunForPlanet();

  const bodies = [
    {
      color: 0x355d72, accent: 0x7d9a64, bands: false, rotationSpeed: 0.1,
      position: new THREE.Vector3(8, 6, -14), radius: 1.1,
    },
    {
      color: 0xb99460, accent: 0xe2cf9d, bands: true, rotationSpeed: 0.045, rings: true,
      position: new THREE.Vector3(-31, 18, -38), radius: 1.45,
    },
    {
      color: 0x9c5140, accent: 0xd98562, bands: false, rotationSpeed: 0.07,
      position: new THREE.Vector3(33, 22, -19), radius: 0.8,
    },
    {
      color: 0x4f777b, accent: 0x8db8b5, bands: true, rotationSpeed: 0.055,
      position: new THREE.Vector3(-34, 9, 29), radius: 0.95,
    },
  ];
  for (const body of bodies) {
    const system = new THREE.Group();
    system.position.copy(body.position);
    system.userData.rotationSpeed = body.rotationSpeed;
    cosmosRoot.add(system);
    const planet = new THREE.Mesh(
      new THREE.SphereGeometry(body.radius, 40, 28),
      new THREE.MeshStandardMaterial({
        map: createPlanetTexture(body.color, body.accent, body.bands),
        roughness: 0.96,
        metalness: 0,
      }),
    );
    planet.userData.rotationSpeed = body.rotationSpeed;
    system.add(planet);
    if (body.rings) {
      const rings = new THREE.Mesh(
        new THREE.RingGeometry(body.radius * 1.35, body.radius * 2.05, 96),
        new THREE.MeshStandardMaterial({
          color: 0xb8a47f,
          roughness: 0.9,
          side: THREE.DoubleSide,
          transparent: true,
          opacity: 0.78,
        }),
      );
      rings.rotation.x = Math.PI / 2.8;
      system.add(rings);
    }
  }
}

function updateSunForPlanet() {
  if (!state.sun) return;
  const distance = SUN_DISTANCES[state.planet] || SUN_DISTANCES.earth;
  const position = camera.position.clone().addScaledVector(sunViewingDirection, distance);
  state.sun.position.copy(position);
  state.sun.scale.setScalar(4.2);
  if (state.sunGlow) {
    state.sunGlow.position.copy(position);
    state.sunGlow.scale.set(15, 15, 1);
  }
  keyLight.position.copy(position);
}

function createGlowTexture() {
  const canvas = document.createElement("canvas");
  canvas.width = 128;
  canvas.height = 128;
  const context = canvas.getContext("2d");
  if (!context) throw new Error("No se pudo crear el resplandor del Sol.");
  const gradient = context.createRadialGradient(64, 64, 5, 64, 64, 64);
  gradient.addColorStop(0, "rgba(255,255,255,0.95)");
  gradient.addColorStop(0.18, "rgba(255,231,173,0.7)");
  gradient.addColorStop(0.45, "rgba(255,190,92,0.22)");
  gradient.addColorStop(1, "rgba(255,160,50,0)");
  context.fillStyle = gradient;
  context.fillRect(0, 0, 128, 128);
  return new THREE.CanvasTexture(canvas);
}

function createPlanetTexture(baseColor, accentColor, banded) {
  const width = 512;
  const height = 256;
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d");
  if (!context) throw new Error("No se pudo crear la textura de los planetas.");

  const base = [
    (baseColor >> 16) & 0xff,
    (baseColor >> 8) & 0xff,
    baseColor & 0xff,
  ];
  const accent = [
    (accentColor >> 16) & 0xff,
    (accentColor >> 8) & 0xff,
    accentColor & 0xff,
  ];
  const image = context.createImageData(width, height);
  for (let y = 0; y < height; y += 1) {
    const latitude = y / height;
    for (let x = 0; x < width; x += 1) {
      const longitude = x / width;
      const band = banded
        ? 0.12 + 0.22 * (0.5 + 0.5 * Math.sin(latitude * Math.PI * 24 + Math.sin(longitude * Math.PI * 4) * 0.3))
        : 0.075 * (0.5 + 0.5 * Math.sin(
          latitude * Math.PI * 9
          + Math.sin(longitude * Math.PI * 8) * 1.4
          + Math.cos(longitude * Math.PI * 5 + latitude * Math.PI * 4),
        ));
      const noise = 0.94
        + 0.06 * Math.sin(x * 12.9898 + y * 78.233) * Math.sin(y * 4.1414 + x * 0.317);
      const offset = (y * width + x) * 4;
      for (let channel = 0; channel < 3; channel += 1) {
        const value = base[channel] * (1 - band) + accent[channel] * band;
        image.data[offset + channel] = Math.max(0, Math.min(255, value * noise));
      }
      image.data[offset + 3] = 255;
    }
  }
  context.putImageData(image, 0, 0);

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.anisotropy = Math.min(4, renderer.capabilities.getMaxAnisotropy());
  return texture;
}

function spawnShootingStar() {
  const start = new THREE.Vector3(
    THREE.MathUtils.randFloat(-32, 32),
    THREE.MathUtils.randFloat(18, 38),
    THREE.MathUtils.randFloat(-40, 20),
  );
  const velocity = new THREE.Vector3(
    THREE.MathUtils.randFloat(-12, 12),
    THREE.MathUtils.randFloat(-9, -5),
    THREE.MathUtils.randFloat(2, 8),
  );
  const group = new THREE.Group();
  const head = new THREE.Mesh(
    new THREE.SphereGeometry(0.09, 8, 8),
    new THREE.MeshBasicMaterial({ color: 0xffffff }),
  );
  group.add(head);
  const tailGeometry = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(0, 0, 0),
    new THREE.Vector3(-velocity.x * 0.12, -velocity.y * 0.12, -velocity.z * 0.12),
  ]);
  const tail = new THREE.Line(
    tailGeometry,
    new THREE.LineBasicMaterial({ color: 0xd9edff, transparent: true, opacity: 0.85 }),
  );
  group.add(tail);
  group.position.copy(start);
  cosmosRoot.add(group);
  state.meteors.push({
    group, velocity, startedAt: performance.now(), duration: 1800,
  });
}

function updateCosmos(now) {
  if (now > state.meteorTimer) {
    spawnShootingStar();
    state.meteorTimer = now + THREE.MathUtils.randInt(6500, 14500);
  }
  for (let index = state.meteors.length - 1; index >= 0; index -= 1) {
    const meteor = state.meteors[index];
    const age = now - meteor.startedAt;
    const seconds = Math.min(age, meteor.duration) / 1000;
    meteor.group.position.addScaledVector(
      meteor.velocity,
      Math.max(0, seconds - (meteor.lastSeconds ?? seconds)),
    );
    meteor.lastSeconds = seconds;
    const opacity = Math.max(0, 1 - age / meteor.duration);
    meteor.group.children[1].material.opacity = opacity * 0.85;
    if (age >= meteor.duration) {
      cosmosRoot.remove(meteor.group);
      meteor.group.traverse((child) => {
        if (child.geometry) child.geometry.dispose();
        if (child.material) child.material.dispose();
      });
      state.meteors.splice(index, 1);
    }
  }
  for (const body of cosmosRoot.children) {
    if (body.userData.rotationSpeed) {
      body.rotation.y += body.userData.rotationSpeed * 0.001;
    }
  }
  if (state.planetVisual) state.planetVisual.rotation.y += 0.0007;
  if (state.moonSystem) state.moonSystem.rotation.y += 0.0012;
}

function mesh(geometry, material, position) {
  const object = new THREE.Mesh(geometry, material);
  if (position) object.position.copy(position);
  object.castShadow = true;
  object.receiveShadow = true;
  return object;
}

function createObjectGroup(kind, cell, id) {
  const group = new THREE.Group();
  group.userData = { role: "map-object", id, kind };

  if (kind === "rock") {
    const radius = 0.28 + Math.random() * 0.12;
    const geometry = new THREE.IcosahedronGeometry(radius, 2);
    const vertices = geometry.attributes.position;
    const seed = id.length * 17;
    for (let index = 0; index < vertices.count; index += 1) {
      const x = vertices.getX(index);
      const y = vertices.getY(index);
      const z = vertices.getZ(index);
      const variation = 0.82 + hashNoise(
        Math.round(x * 1000) + seed,
        Math.round(y * 1000) + Math.round(z * 1000),
      ) * 0.34;
      vertices.setXYZ(index, x * variation, y * variation, z * variation);
    }
    geometry.computeVertexNormals();
    const rockMaterials = [materials.rock, materials.rockWarm, materials.rockCool];
    const stone = mesh(
      geometry,
      rockMaterials[(seed + id.charCodeAt(id.length - 1)) % rockMaterials.length],
    );
    stone.scale.set(1.05, 0.72 + Math.random() * 0.42, 0.9);
    stone.rotation.set(Math.random() * 0.4, Math.random() * Math.PI, Math.random() * 0.4);
    group.add(stone);
    for (let pebble = 0; pebble < 2; pebble += 1) {
      const smallStone = mesh(
        new THREE.DodecahedronGeometry(0.09 + Math.random() * 0.07, 0),
        materials.pebble,
        new THREE.Vector3(
          (Math.random() - 0.5) * 0.72,
          0.035,
          (Math.random() - 0.5) * 0.62,
        ),
      );
      smallStone.rotation.set(Math.random(), Math.random(), Math.random());
      group.add(smallStone);
    }
  } else if (kind === "crater") {
    const disc = mesh(
      createCraterBasin(),
      materials.craterFloor,
    );
    disc.position.y = 0.01;
    group.add(disc);
    const rim = mesh(
      new THREE.TorusGeometry(0.47, 0.085, 7, 40),
      materials.craterRim,
    );
    rim.rotation.x = Math.PI / 2;
    rim.position.y = 0.045;
    rim.scale.set(1.08, 0.92, 0.78);
    group.add(rim);
    for (let fragment = 0; fragment < 5; fragment += 1) {
      const angle = (fragment / 5) * Math.PI * 2 + Math.random() * 0.3;
      const stone = mesh(
        new THREE.DodecahedronGeometry(0.07 + Math.random() * 0.08, 0),
        fragment % 2 ? materials.rockWarm : materials.rock,
        new THREE.Vector3(
          Math.cos(angle) * (0.58 + Math.random() * 0.13),
          0.025,
          Math.sin(angle) * (0.58 + Math.random() * 0.13),
        ),
      );
      stone.rotation.set(Math.random(), Math.random(), Math.random());
      group.add(stone);
    }
  } else if (kind === "gas") {
    const floor = mesh(
      new THREE.CircleGeometry(1.6, 48),
      materials.gasGround,
    );
    floor.rotation.x = -Math.PI / 2;
    floor.position.y = 0.035;
    floor.scale.set(1.1, 0.82, 1);
    group.add(floor);
    const clouds = [
      { position: [0, 0.35, 0], scale: [1.05, 0.56, 0.9] },
      { position: [-0.38, 0.28, 0.2], scale: [0.62, 0.43, 0.62] },
      { position: [0.34, 0.42, -0.25], scale: [0.72, 0.52, 0.58] },
      { position: [0.12, 0.62, 0.25], scale: [0.48, 0.38, 0.5] },
      { position: [-0.1, 0.2, -0.47], scale: [0.56, 0.32, 0.5] },
    ];
    for (const puff of clouds) {
      const cloud = mesh(
        new THREE.SphereGeometry(0.72, 20, 14),
        materials.gas,
        new THREE.Vector3(...puff.position),
      );
      cloud.scale.set(...puff.scale);
      group.add(cloud);
    }
  } else if (kind === "rough") {
    for (let i = 0; i < 5; i += 1) {
      const stone = mesh(
        new THREE.TetrahedronGeometry(0.18 + Math.random() * 0.13, 0),
        materials.rough,
        new THREE.Vector3((Math.random() - 0.5) * 0.65, 0.12, (Math.random() - 0.5) * 0.65),
      );
      stone.rotation.set(Math.random(), Math.random(), Math.random());
      group.add(stone);
    }
  } else if (kind === "cliff") {
    const pit = mesh(
      new THREE.CircleGeometry(0.45, 32),
      materials.cliffFloor,
      new THREE.Vector3(0, -0.12, 0),
    );
    pit.rotation.x = -Math.PI / 2;
    pit.scale.set(0.72, 1.2, 1);
    group.add(pit);
    for (const side of [-1, 1]) {
      const ledge = mesh(
        createCliffWall(side),
        materials.cliffWall,
      );
      group.add(ledge);
      const strata = mesh(
        new THREE.DodecahedronGeometry(0.24, 1),
        materials.cliffStrata,
        new THREE.Vector3(side * 0.38, 0.13, (Math.random() - 0.5) * 0.56),
      );
      strata.scale.set(0.65, 0.48, 1.55);
      group.add(strata);
    }
    group.rotation.y = Math.random() < 0.5 ? 0 : Math.PI / 2;
  } else if (kind === "science") {
    const shape = new THREE.Shape();
    for (let point = 0; point < 10; point += 1) {
      const radius = point % 2 === 0 ? 0.28 : 0.13;
      const angle = point * Math.PI / 5 - Math.PI / 2;
      const x = Math.cos(angle) * radius;
      const y = Math.sin(angle) * radius;
      if (point === 0) shape.moveTo(x, y);
      else shape.lineTo(x, y);
    }
    shape.closePath();
    const marker = mesh(
      new THREE.ShapeGeometry(shape),
      new THREE.MeshStandardMaterial({
        color: 0x9bd8f1, emissive: 0x245d80, emissiveIntensity: 0.4,
        side: THREE.DoubleSide,
      }),
      new THREE.Vector3(0, 0.18, 0),
    );
    marker.rotation.x = -Math.PI / 2;
    group.add(marker);
  }

  group.position.copy(cellWorldPosition(cell));
  return group;
}

async function addObject(kind, cell) {
  if ([state.start, state.goal, state.rover].some((point) => sameCell(point, cell))) {
    showToast("No se puede colocar un objeto sobre el rover o un punto de misión.");
    return;
  }
  const id = `operator-${++state.objectCounter}`;
  const data = await postAndSync("/api/objects", {
    id, kind, row: cell.row, col: cell.col,
  });
  if (data) showToast(`${kindLabel(kind)} añadido · F${cell.row + 1} C${cell.col + 1}`);
}

function removeVisualObject(id) {
  const entity = state.objects.get(id);
  if (!entity) return;
  worldRoot.remove(entity.group);
  entity.group.traverse((child) => {
    if (child.isMesh && child.geometry) child.geometry.dispose();
  });
  state.objects.delete(id);
}

function kindLabel(kind) {
  return ({
    rock: "Piedra", crater: "Cráter", gas: "Zona de gas", rough: "Terreno rocoso",
    cliff: "Barranco", science: "Lugar científico",
  })[kind] || kind;
}

function renderScienceSites(sites, objects) {
  while (scienceRoot.children.length) {
    const child = scienceRoot.children.pop();
    scienceRoot.remove(child);
    child.traverse((part) => {
      if (part.geometry) part.geometry.dispose();
    });
  }
  const represented = new Set(
    (objects || []).filter((item) => item.kind === "science")
      .map((item) => `${item.row},${item.col}`),
  );
  for (const site of sites || []) {
    if (represented.has(`${site.row},${site.col}`)) continue;
    const marker = createObjectGroup("science", site, `science-${site.row}-${site.col}`);
    marker.userData.role = "science-marker";
    marker.userData.cell = { row: site.row, col: site.col };
    scienceRoot.add(marker);
  }
}

async function apiRequest(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: {
      ...(options.body ? { "Content-Type": "application/json" } : {}),
      ...options.headers,
    },
  });
  let payload;
  try {
    payload = await response.json();
  } catch {
    throw new Error(`El servidor respondió con un formato inesperado (${response.status}).`);
  }
  if (!response.ok) throw new Error(payload.error || `Error HTTP ${response.status}`);
  return payload;
}

function renderBeliefs(beliefs) {
  while (beliefRoot.children.length) {
    const child = beliefRoot.children.pop();
    beliefRoot.remove(child);
    child.geometry?.dispose();
    child.material?.dispose();
  }
  for (const belief of beliefs || []) {
    if (belief.risk < 0.3 || belief.kind === "libre") continue;
    const position = cellPosition(belief);
    const material = new THREE.MeshBasicMaterial({
      color: belief.blocked ? 0xed6f61 : 0xe7be68,
      transparent: true,
      opacity: Math.min(0.35, 0.09 + belief.risk * 0.22),
      depthWrite: false,
    });
    const tile = new THREE.Mesh(new THREE.PlaneGeometry(0.82, 0.82), material);
    tile.rotation.x = -Math.PI / 2;
    tile.position.set(position.x, terrainHeight(position.x, position.z) + 0.06, position.z);
    beliefRoot.add(tile);
  }
}

function syncFromPython(data) {
  if (!data || typeof data !== "object") throw new Error("El estado de misión recibido está vacío.");
  const planetChanged = state.planet !== data.planet;
  state.pythonState = data;
  state.planet = data.planet;
  if (planetChanged && state.terrain) {
    createTerrain();
    if (state.roverGroup) worldRoot.remove(state.roverGroup);
    state.roverGroup = createRover();
    state.roverFrom = null;
  }
  state.start = { ...data.start };
  state.goal = { ...data.goal };
  state.rover = { ...data.rover };
  state.energy = data.battery;
  state.steps = data.steps;
  state.strategy = data.strategy;

  const incoming = new Map((data.objects || []).map((item) => [item.id, item]));
  for (const id of [...state.objects.keys()]) {
    if (!incoming.has(id)) removeVisualObject(id);
  }
  for (const item of incoming.values()) {
    const cell = { row: item.row, col: item.col };
    const current = state.objects.get(item.id);
    if (!current || current.kind !== item.kind) {
      if (current) removeVisualObject(item.id);
      const group = createObjectGroup(item.kind, cell, item.id);
      state.objects.set(item.id, { id: item.id, kind: item.kind, cell, group });
      worldRoot.add(group);
    } else {
      current.cell = cell;
      updateEntityPosition(current);
    }
  }
  renderScienceSites(data.science_sites, data.objects);
  if (state.startMarker) updateMarkerPosition(state.startMarker, state.start);
  if (state.goalMarker) updateMarkerPosition(state.goalMarker, state.goal);
  if (state.roverGroup) updateRoverTarget();

  const selectedCells = data.routes?.selected || [];
  state.activeRoute = {
    cells: selectedCells,
    distance: Math.max(0, selectedCells.length - 1) * CELL_METERS,
    risk: selectedCells.slice(1).reduce(
      (sum, cell) => sum + Number(data.beliefs?.find(
        (belief) => belief.row === cell.row && belief.col === cell.col,
      )?.risk || 0.7),
      0,
    ),
  };
  renderBeliefs(data.beliefs);
  el("planet").value = data.planet;
  el("strategy").value = data.strategy;
  el("riskWeight").value = String(data.risk_weight);
  el("riskValue").textContent = Number(data.risk_weight).toFixed(2);
  el("gasLevel").value = String(Math.round(data.gas_concentration * 100));
  el("gasValue").textContent = `${Math.round(data.gas_concentration * 100)}%`;
  el("battery").value = String(Math.round(data.battery_initial));
  el("batteryValue").textContent = `${Math.round(data.battery)}%`;
  el("energyStat").innerHTML = `${Math.round(data.battery)}<small>%</small>`;
  el("sciencePriority").value = String(Math.round(data.science_priority * 100));
  el("sciencePriorityValue").textContent = `${Math.round(data.science_priority * 100)}%`;
  el("planetName").textContent = data.planet_profile?.name?.toUpperCase() || data.planet.toUpperCase();
  el("modeName").textContent = profileModeLabel(data);
  el("roverLegend").textContent = data.planet_profile?.mode === "sonda_virtual" ? "SONDA" : "ROVER";
  el("planetNotice").textContent = data.planet_profile?.mode === "sonda_virtual"
    ? `${data.planet_profile.name}: sonda virtual; la cuadrícula es una abstracción y no representa suelo sólido.`
    : `${data.planet_profile?.name || "Planeta"} · parámetros y condiciones de simulación pedagógicos.`;
  const gasGiant = data.planet_profile?.mode === "sonda_virtual";
  el("planetNotice").classList.toggle("warning", gasGiant);
  for (const button of document.querySelectorAll(".tool")) {
    button.disabled = gasGiant && ["rock", "crater", "rough"].includes(button.dataset.tool);
  }

  const profile = PROFILES[state.planet];
  if (profile) {
    scene.background = new THREE.Color(profile.background);
    scene.fog = new THREE.Fog(profile.background, 80, 175);
  }
  el("positionStat").textContent = `F${String(data.rover.row + 1).padStart(2, "0")} · C${String(data.rover.col + 1).padStart(2, "0")}`;
  el("coordX").textContent = String(data.rover.col + 1).padStart(2, "0");
  el("coordZ").textContent = String(data.rover.row + 1).padStart(2, "0");
  el("stepStat").textContent = `PASOS: ${data.steps}`;
  el("scienceStat").textContent = String(data.science_collected);
  if (data.mission_complete || data.failure) {
    stopMission(data.failure || "Destino alcanzado. Misión completada.", data.failure ? "error" : "ready");
  }
  refreshRoutes();
  renderDiagnostics(data);
}

function profileModeLabel(data) {
  const mode = data.planet_profile?.mode;
  return mode === "sonda_virtual"
    ? "SONDA ATMOSFÉRICA · MAPA VIRTUAL"
    : mode === "rover_hipotetico"
      ? "ROVER HIPOTÉTICO · TERRENO SIMULADO"
      : "TERRENO DE EXPLORACIÓN";
}

function renderDiagnostics(data) {
  const decision = data.last_decision;
  const metrics = data.perception_metrics || {};
  const test = metrics.metrics?.test;
  const validation = metrics.metrics?.validation;
  const train = metrics.metrics?.train;
  if (test) {
    const trainingStatus = metrics.converged
      ? `convergió en ${metrics.training_iterations} iteraciones`
      : `llegó al límite de ${metrics.training_iterations} iteraciones`;
    el("rnaMetrics").textContent =
      `MLP 8–32–16–3 · escenarios separados ` +
      `train/val/test: F1 ${Math.round((train?.macro_f1 || 0) * 100)}/` +
      `${Math.round((validation?.macro_f1 || 0) * 100)}/` +
      `${Math.round(test.macro_f1 * 100)}% · ${trainingStatus} · ` +
      `recall barranco ${Math.round(test.hazard_recall * 100)}% · ` +
      `matriz [${metrics.confusion_matrix.values.map((row) => row.join(",")).join(" / ")}]`;
  }
  renderLineChart("rnaLossChart", metrics.loss_curve || [], false);
  if (decision) {
    el("decisionBadge").textContent = `PASO ${decision.step} · ${decision.supervisor.action.type}`;
    el("rnaPrediction").textContent = `Predicción: ${decision.rna.prediction}`;
    el("rnaConfidence").textContent = `Confianza: ${(decision.rna.confidence * 100).toFixed(1)}% · entrada de 8 sensores`;
    const probabilityNames = { libre: "Libre", obstaculo: "Obstáculo", barranco: "Barranco" };
    el("rnaProbabilities").innerHTML = Object.entries(decision.rna.probabilities)
      .map(([key, value]) => `<div class="probability-row"><span>${probabilityNames[key] || key}</span><div class="probability-track"><div class="probability-fill" style="width:${Math.max(0, Math.min(100, value * 100))}%"></div></div><b>${(value * 100).toFixed(0)}%</b></div>`)
      .join("");
    const sensorLabels = {
      distancia_obstaculo_m: ["Proximidad obstáculo", "m"],
      distancia_crater_m: ["Altimetría / cráter", "m"],
      pendiente_grados: ["Pendiente", "°"],
      vibracion_normalizada: ["Vibración", "0–1"],
      textura_roca: ["Textura", "0–1"],
      temperatura_normalizada: ["Temperatura normalizada", "0–1"],
      radiacion_normalizada: ["Radiación normalizada", "0–1"],
      gas_normalizado: ["Gas", "0–1"],
    };
    el("rnaSensors").innerHTML = Object.entries(decision.sensors.features)
      .map(([key, value]) => {
        const [label, unit] = sensorLabels[key] || [key, ""];
        return `<div class="sensor-reading"><span>${label}</span><b>${Number(value).toFixed(2)} ${unit}</b></div>`;
      })
      .join("");
    const fuzzy = decision.fuzzy;
    el("fuzzyOutput").textContent = `Velocidad ${fuzzy.speed.toFixed(2)} · cautela ${fuzzy.caution.toFixed(2)}`;
    el("fuzzyMethod").textContent = `${fuzzy.defuzzification} · áreas agregadas ${fuzzy.aggregation_area}`;
    const membershipNames = {
      bateria_critica: "Batería crítica", bateria_baja: "Batería baja",
      bateria_media: "Batería media", bateria_alta: "Batería alta",
      riesgo_cerca: "Riesgo alto", riesgo_medio: "Riesgo medio",
      riesgo_bajo: "Riesgo bajo", pendiente_fuerte: "Pendiente fuerte",
      gas_alto: "Gas alto", confianza_baja: "Confianza baja",
    };
    el("fuzzyMemberships").innerHTML = Object.entries(fuzzy.memberships || {})
      .filter(([, value]) => value > 0.01)
      .map(([key, value]) => `<div class="membership-row"><span>${membershipNames[key] || key}</span><b>${value.toFixed(2)}</b></div>`)
      .join("") || '<div class="empty-note">Ningún conjunto lingüístico activado.</div>';
    el("fuzzyRules").innerHTML = fuzzy.rules?.length
      ? fuzzy.rules.map((rule) => `<div class="rule-item"><span>${rule.id} · ${Math.round(rule.activation * 100)}%</span>${rule.text}</div>`).join("")
      : '<div class="empty-note">No se dispararon reglas.</div>';
  }

  const selectedKey = JSON.stringify(data.routes?.selected || []);
  const routes = (data.routes?.candidates || []).map((route) => ({
    ...route,
    selected: JSON.stringify(route.cells) === selectedKey,
  }));
  el("generationCount").textContent = `${data.routes?.history?.length || 0} GENERACIONES`;
  el("candidateList").innerHTML = routes.length
    ? routes.slice(0, 3).map((route) => `<div class="candidate${route.selected ? " best" : ""}"><div class="candidate-name">${route.name}${route.selected ? " · SELECCIONADA" : ""}</div><div class="candidate-metrics"><span>Distancia <b>${route.distance_m.toFixed(0)} m</b></span><span>Riesgo <b>${route.risk.toFixed(2)}</b></span><span>Energía <b>${route.energy.toFixed(1)}%</b></span><span>Fitness <b>${route.fitness.toFixed(3)}</b></span></div></div>`).join("")
    : '<div class="empty-note">Sin ruta candidata factible.</div>';
  renderFitnessChart(data.routes?.history || []);
  const events = data.events || [];
  el("logCount").textContent = `${events.length} EVENTOS`;
  const missionLog = el("missionLog");
  missionLog.replaceChildren();
  if (events.length) {
    for (const event of [...events].reverse().slice(0, 16)) {
      const row = document.createElement("div");
      row.className = "log-row";
      const step = document.createElement("time");
      step.textContent = `PASO ${event.step}`;
      const type = document.createElement("span");
      type.className = "log-type";
      type.textContent = event.type;
      const message = document.createElement("span");
      message.textContent = event.message;
      row.append(step, type, message);
      missionLog.append(row);
    }
  } else {
    const empty = document.createElement("div");
    empty.className = "empty-note";
    empty.textContent = "Las observaciones, decisiones, eventos y replanteos aparecerán aquí.";
    missionLog.append(empty);
  }
}

function renderFitnessChart(history) {
  const svg = el("fitnessChart");
  if (!history.length) {
    svg.innerHTML = "";
    return;
  }
  renderLineChart(
    "fitnessChart",
    history.map((entry) => Number(entry.best_fitness)),
    true,
  );
}

function renderLineChart(id, values, lowerIsBetter) {
  const svg = el(id);
  if (!values.length) {
    svg.innerHTML = "";
    return;
  }
  const numericValues = values.map((entry) => Number(
    typeof entry === "object" ? entry.best_fitness : entry,
  ));
  const min = Math.min(...numericValues);
  const max = Math.max(...numericValues);
  const span = Math.max(0.001, max - min);
  const points = numericValues.map((value, index) => {
    const ratio = span === 0 ? 0.5 : (value - min) / span;
    const x = 14 + (index / Math.max(1, numericValues.length - 1)) * 610;
    const y = 81 - (lowerIsBetter ? 1 - ratio : ratio) * 62;
    return [x, y];
  });
  const pointList = points.map(([x, y]) => `${x},${y}`).join(" ");
  svg.innerHTML = `<path class="chart-grid" d="M12 20H628 M12 50H628 M12 82H628"/><polyline class="chart-line" points="${pointList}"/>${points.length <= 40 ? points.map(([x, y]) => `<circle class="chart-point" cx="${x}" cy="${y}" r="2.1"/>`).join("") : ""}`;
}

async function bootstrapEngine() {
  setStatus("Cargando motor Python, entrenando la RNA y preparando la misión.");
  try {
    const data = await apiRequest("/api/state");
    const isGiant = PROFILES[data.planet]?.gasGiant;
    state.planet = data.planet;
    createTerrain();
    syncFromPython(data);
    setStatus("Motor Python conectado · RNA, lógica difusa y AG listos.");
  } catch (error) {
    setStatus(`No se pudo conectar al motor Python: ${error.message}`, "error");
    showToast("Abre la aplicación con: py -3 ares_x_server.py");
  }
}

async function postAndSync(path, body) {
  try {
    const data = await apiRequest(path, {
      method: "POST",
      body: JSON.stringify(body),
    });
    syncFromPython(data);
    return data;
  } catch (error) {
    setStatus(error.message, "error");
    showToast(error.message);
    return null;
  }
}

function makeMarker(role) {
  const group = new THREE.Group();
  group.userData = { role };
  if (role === "start") {
    const base = mesh(
      new THREE.CylinderGeometry(0.22, 0.22, 0.08, 16),
      new THREE.MeshStandardMaterial({ color: 0x5da8ff, emissive: 0x173967 }),
    );
    base.position.y = 0.09;
    group.add(base);
    const beacon = mesh(
      new THREE.ConeGeometry(0.13, 0.32, 12),
      new THREE.MeshStandardMaterial({ color: 0x8fc2ff, emissive: 0x214e85 }),
      new THREE.Vector3(0, 0.31, 0),
    );
    group.add(beacon);
  } else {
    const starShape = new THREE.Shape();
    for (let point = 0; point < 10; point += 1) {
      const radius = point % 2 === 0 ? 0.42 : 0.19;
      const angle = point * Math.PI / 5 - Math.PI / 2;
      const x = Math.cos(angle) * radius;
      const y = Math.sin(angle) * radius;
      if (point === 0) starShape.moveTo(x, y);
      else starShape.lineTo(x, y);
    }
    starShape.closePath();
    const star = mesh(
      new THREE.ShapeGeometry(starShape),
      new THREE.MeshStandardMaterial({
        color: 0xffd26f, emissive: 0xb87920, emissiveIntensity: 0.55,
        side: THREE.DoubleSide,
      }),
      new THREE.Vector3(0, 0.13, 0),
    );
    star.rotation.x = -Math.PI / 2;
    group.add(star);
    const aura = mesh(
      new THREE.TorusGeometry(0.55, 0.018, 6, 48),
      new THREE.MeshBasicMaterial({ color: 0xffd26f, transparent: true, opacity: 0.6 }),
      new THREE.Vector3(0, 0.075, 0),
    );
    aura.rotation.x = Math.PI / 2;
    group.add(aura);
    const beacon = mesh(
      new THREE.CylinderGeometry(0.025, 0.025, 0.7, 8),
      new THREE.MeshBasicMaterial({ color: 0xf1bd62, transparent: true, opacity: 0.62 }),
      new THREE.Vector3(0, 0.39, 0),
    );
    group.add(beacon);
  }
  worldRoot.add(group);
  return group;
}

function updateMarkerPosition(marker, cell) {
  marker.position.copy(cellWorldPosition(cell));
}

function createRover() {
  const rover = new THREE.Group();
  rover.userData = { role: "rover" };
  if (PROFILES[state.planet].gasGiant) {
    const shell = mesh(
      new THREE.SphereGeometry(0.3, 18, 14),
      new THREE.MeshStandardMaterial({ color: 0xe9ba70, metalness: 0.3, roughness: 0.44 }),
      new THREE.Vector3(0, 0.5, 0),
    );
    shell.scale.set(0.82, 1.15, 0.82);
    rover.add(shell);
    const nose = mesh(
      new THREE.ConeGeometry(0.16, 0.38, 12),
      new THREE.MeshStandardMaterial({ color: 0x86beca, metalness: 0.24, roughness: 0.42 }),
      new THREE.Vector3(0, 0.88, 0),
    );
    rover.add(nose);
    const antenna = mesh(
      new THREE.CylinderGeometry(0.018, 0.018, 0.56, 6),
      new THREE.MeshStandardMaterial({ color: 0xcbd1c4, metalness: 0.4 }),
      new THREE.Vector3(0.25, 0.56, 0),
    );
    antenna.rotation.z = Math.PI / 3;
    rover.add(antenna);
    worldRoot.add(rover);
    return rover;
  }

  const bodyMaterial = new THREE.MeshStandardMaterial({
    color: 0xe97755, roughness: 0.54, metalness: 0.16,
  });
  const darkMaterial = new THREE.MeshStandardMaterial({ color: 0x252a29, roughness: 0.86 });
  const body = mesh(new THREE.BoxGeometry(0.8, 0.28, 0.62), bodyMaterial);
  body.position.y = 0.48;
  rover.add(body);
  const cabin = mesh(
    new THREE.BoxGeometry(0.37, 0.2, 0.42),
    new THREE.MeshStandardMaterial({ color: 0xd4d8c9, metalness: 0.25, roughness: 0.46 }),
    new THREE.Vector3(-0.02, 0.72, 0),
  );
  rover.add(cabin);
  const sensor = mesh(
    new THREE.SphereGeometry(0.1, 12, 10),
    new THREE.MeshStandardMaterial({ color: 0x8bd1da, emissive: 0x17464b }),
    new THREE.Vector3(0.25, 0.85, 0),
  );
  rover.add(sensor);
  const solarMaterial = new THREE.MeshStandardMaterial({
    color: 0x294e66, metalness: 0.52, roughness: 0.36, side: THREE.DoubleSide,
  });
  for (const side of [-1, 1]) {
    const panel = mesh(
      new THREE.BoxGeometry(0.72, 0.035, 0.37),
      solarMaterial,
      new THREE.Vector3(-0.02, 0.47, side * 0.65),
    );
    panel.rotation.x = side * 0.035;
    rover.add(panel);
    const strut = mesh(
      new THREE.CylinderGeometry(0.018, 0.018, 0.42, 6),
      darkMaterial,
      new THREE.Vector3(-0.02, 0.34, side * 0.48),
    );
    strut.rotation.x = Math.PI / 2;
    rover.add(strut);
  }
  const mast = mesh(
    new THREE.CylinderGeometry(0.025, 0.025, 0.5, 8),
    new THREE.MeshStandardMaterial({ color: 0xb9c2b9, metalness: 0.55 }),
    new THREE.Vector3(-0.26, 0.82, 0),
  );
  rover.add(mast);
  const dish = mesh(
    new THREE.SphereGeometry(0.16, 16, 10, 0, Math.PI * 2, 0, Math.PI / 2),
    new THREE.MeshStandardMaterial({ color: 0xd5cfb6, metalness: 0.46, roughness: 0.4 }),
    new THREE.Vector3(-0.26, 1.08, 0),
  );
  dish.rotation.x = -Math.PI / 2;
  rover.add(dish);
  for (const x of [-0.29, 0.29]) {
    for (const z of [-0.38, 0.38]) {
      const wheel = mesh(
        new THREE.CylinderGeometry(0.16, 0.16, 0.12, 12),
        darkMaterial,
        new THREE.Vector3(x, 0.29, z),
      );
      wheel.rotation.z = Math.PI / 2;
      rover.add(wheel);
    }
  }
  worldRoot.add(rover);
  return rover;
}

function updateRoverTarget() {
  if (!state.roverGroup) return;
  const target = cellWorldPosition(state.rover, 0.04);
  if (!state.roverFrom) {
    state.roverGroup.position.copy(target);
    return;
  }
  state.roverTo = target;
}

function objectsAt(cell) {
  return [...state.objects.values()].filter((item) => sameCell(item.cell, cell));
}

function isBlocked(cell, ignoreId = null) {
  return objectsAt(cell).some(
    (item) => item.id !== ignoreId
      && ["rock", "crater", "cliff", "meteorite"].includes(item.kind),
  );
}

function riskAt(cell) {
  let risk = 0;
  const concentration = Number(el("gasLevel").value) / 70;
  for (const item of state.objects.values()) {
    const distance = distanceCells(cell, item.cell);
    if (item.kind === "rough" && distance === 0) risk += 2.4;
    if (item.kind === "crater") {
      if (distance === 1) risk += 8;
      else if (distance === 2) risk += 3;
    }
    if (item.kind === "gas" && distance <= 2) {
      risk += (3 - distance) * 2.4 * concentration;
    }
  }
  return risk;
}

function findRoute(riskWeight) {
  if (isBlocked(state.rover) || isBlocked(state.goal)) return null;
  const startKey = cellKey(state.rover);
  const goalKey = cellKey(state.goal);
  const frontier = new MinHeap();
  frontier.push({ cost: 0, cell: { ...state.rover } });
  const costs = new Map([[startKey, 0]]);
  const previous = new Map();

  while (frontier.length) {
    const currentItem = frontier.pop();
    if (!currentItem) break;
    const { cost, cell } = currentItem;
    const key = cellKey(cell);
    if (cost > (costs.get(key) ?? Infinity)) continue;
    if (key === goalKey) break;
    for (const [dr, dc] of MOVES) {
      const next = { row: cell.row + dr, col: cell.col + dc };
      if (!inBounds(next.row, next.col) || isBlocked(next)) continue;
      const nextKey = cellKey(next);
      const nextCost = cost + 1 + riskWeight * riskAt(next);
      if (nextCost < (costs.get(nextKey) ?? Infinity)) {
        costs.set(nextKey, nextCost);
        previous.set(nextKey, key);
        frontier.push({ cost: nextCost, cell: next });
      }
    }
  }

  if (!costs.has(goalKey)) return null;
  const cells = [{ ...state.goal }];
  let cursor = goalKey;
  while (cursor !== startKey) {
    const parent = previous.get(cursor);
    if (!parent) return null;
    const [row, col] = parent.split(",").map(Number);
    cells.push({ row, col });
    cursor = parent;
  }
  cells.reverse();
  const risk = cells.slice(1).reduce((sum, cell) => sum + riskAt(cell), 0);
  return {
    cells,
    distance: (cells.length - 1) * CELL_METERS,
    risk,
    score: costs.get(goalKey),
  };
}

function clearRoutes() {
  while (routeRoot.children.length) {
    const child = routeRoot.children.pop();
    if (!child) break;
    routeRoot.remove(child);
    child.geometry?.dispose();
    child.material?.dispose();
  }
}

function drawRoute(route, color, opacity) {
  if (!route || route.length < 2) return;
  const points = route.map((cell) => cellWorldPosition({
    row: Array.isArray(cell) ? cell[0] : cell.row,
    col: Array.isArray(cell) ? cell[1] : cell.col,
  }, 0.14));
  const geometry = new THREE.BufferGeometry().setFromPoints(points);
  const line = new THREE.Line(
    geometry,
    new THREE.LineBasicMaterial({ color, transparent: true, opacity, depthTest: true }),
  );
  routeRoot.add(line);
}

function refreshRoutes() {
  clearRoutes();
  const data = state.pythonState;
  if (!data) return;
  const candidates = data.routes?.candidates || [];
  const shortest = candidates.find((candidate) => candidate.name === "Menor distancia");
  const safe = candidates.find((candidate) => candidate.name === "Menor riesgo");
  const selected = data.routes?.selected || [];
  if (shortest) drawRoute(shortest.cells, 0x71b5e1, 0.75);
  if (safe) drawRoute(safe.cells, 0xc1e879, 0.78);
  drawRoute(selected, 0xffe178, 1);
  if (selected.length > 1) {
    const activeDistance = (selected.length - 1) * CELL_METERS;
    state.activeRoute = {
      cells: selected,
      distance: activeDistance,
      risk: selected.slice(1).reduce(
        (sum, cell) => sum + Number(data.beliefs?.find(
          (belief) => belief.row === cell.row && belief.col === cell.col,
        )?.risk || 0.7),
        0,
      ),
    };
  } else {
    state.activeRoute = null;
  }
  state.distanceRoute = shortest;
  state.safeRoute = safe;
  updateStats();
}

function updateStats() {
  const active = state.activeRoute;
  const data = state.pythonState;
  el("positionStat").textContent = `F${String(state.rover.row + 1).padStart(2, "0")} · C${String(state.rover.col + 1).padStart(2, "0")}`;
  el("coordX").textContent = String(state.rover.col + 1).padStart(2, "0");
  el("coordZ").textContent = String(state.rover.row + 1).padStart(2, "0");
  el("distanceStat").innerHTML = active ? `${active.distance}<small> m</small>` : `— <small>m</small>`;
  el("riskStat").innerHTML = active
    ? `${Math.min(100, Math.round(active.risk * 5))}<small> / 100</small>`
    : `— <small>/ 100</small>`;
  el("energyStat").innerHTML = `${Math.round(state.energy)}<small>%</small>`;
  el("batteryValue").textContent = `${Math.round(state.energy)}%`;
  el("stepStat").textContent = `PASOS: ${state.steps}`;
  el("shortestStat").textContent = state.distanceRoute
    ? `${state.distanceRoute.distance_m.toFixed(0)} m`
    : "Sin ruta";
  el("safeStat").textContent = state.safeRoute
    ? `${state.safeRoute.distance_m.toFixed(0)} m · riesgo ${state.safeRoute.risk.toFixed(1)}`
    : "Sin ruta";
}

function setStatus(message, type = "ready") {
  el("statusText").textContent = message;
  el("statusDot").style.background = type === "error"
    ? "#e77769" : type === "running" ? "#c1e879" : "#86a8b7";
  el("connectionStatus").textContent = type === "running"
    ? "MISIÓN EN CURSO" : type === "error" ? "SIN RUTA DISPONIBLE" : "PLANIFICADOR ACTIVO";
}

function showToast(message) {
  const toast = el("toast");
  toast.textContent = message;
  toast.classList.add("visible");
  clearTimeout(state.toastTimeout);
  state.toastTimeout = setTimeout(() => toast.classList.remove("visible"), 2000);
}

function stopMission(message = "Misión pausada.", type = "ready") {
  state.playing = false;
  clearInterval(state.timer);
  state.timer = null;
  el("play").classList.remove("running");
  el("play").innerHTML = '<span class="play-symbol">▶</span> Iniciar misión';
  if (message) setStatus(message, type);
}

async function advanceOneStep() {
  if (state.stepPending) return false;
  state.stepPending = true;
  const previous = { ...state.rover };
  try {
    const data = await apiRequest("/api/step", { method: "POST" });
    syncFromPython(data);
    if (!sameCell(previous, state.rover)) {
      state.roverFrom = cellWorldPosition(previous, 0.04);
      state.roverTo = cellWorldPosition(state.rover, 0.04);
      state.roverMoveStarted = performance.now();
    }
    if (data.failure) {
      stopMission(data.failure, "error");
      showToast(data.failure);
      return false;
    }
    if (data.mission_complete) {
      stopMission("Destino alcanzado. Misión completada.");
      showToast("Destino alcanzado · misión completada.");
      return false;
    }
    setStatus(
      `RNA + lógica difusa + AG · paso ${data.steps} · posición F${state.rover.row + 1} C${state.rover.col + 1}`,
      state.playing ? "running" : "ready",
    );
    return true;
  } catch (error) {
    stopMission(`Error del motor Python: ${error.message}`, "error");
    showToast(error.message);
    return false;
  } finally {
    state.stepPending = false;
  }
}

async function togglePlay() {
  if (state.playing) {
    stopMission("Misión pausada. Puedes editar el mapa y continuar.");
    return;
  }
  if (sameCell(state.rover, state.goal)) {
    showToast("El rover ya está en el destino. Pulsa Reiniciar.");
    return;
  }
  state.playing = true;
  el("play").classList.add("running");
  el("play").innerHTML = '<span class="play-symbol">Ⅱ</span> Pausar misión';
  setStatus("Misión en curso · motor híbrido Python activo.", "running");
  const moved = await advanceOneStep();
  if (!moved || !state.playing) return;
  state.timer = setInterval(() => { advanceOneStep(); }, Number(el("speed").value));
}

async function resetMission() {
  stopMission(null);
  const data = await postAndSync("/api/reset", {
    seed: state.pythonState?.seed ?? 42,
  });
  if (data) {
    state.roverFrom = null;
    updateRoverTarget();
    setStatus("Misión reiniciada · RNA y planificador Python listos.");
  }
}

async function syncPlanet() {
  if (state.playing) stopMission("Misión pausada por cambio de planeta.");
  state.planet = el("planet").value;
  createTerrain();
  const data = await postAndSync("/api/config", { planet: state.planet });
  if (data) {
    if (state.roverGroup) worldRoot.remove(state.roverGroup);
    state.roverGroup = createRover();
    state.roverFrom = null;
    updateRoverTarget();
  }
}

function syncTool(tool) {
  state.tool = tool;
  for (const button of document.querySelectorAll(".tool")) {
    button.classList.toggle("active", button.dataset.tool === tool);
  }
  const label = tool === "select" ? "Seleccionar" : kindLabel(tool);
  setStatus(tool === "erase"
    ? "Modo borrar: pulsa una celda u objeto para retirarlo."
    : `Modo ${label}: pulsa el terreno para colocar; arrastra elementos existentes para moverlos.`);
}

function cellFromIntersection(intersection) {
  const col = Math.floor(intersection.point.x + SIZE / 2);
  const row = Math.floor(intersection.point.z + SIZE / 2);
  return inBounds(row, col) ? { row, col } : null;
}

function updateEntityPosition(entity) {
  entity.group.position.copy(cellWorldPosition(entity.cell));
}

function pickGroup(intersection) {
  let current = intersection.object;
  while (current && current !== scene) {
    if (current.userData?.role) return current;
    current = current.parent;
  }
  return null;
}

function pickables() {
  return [
    ...[...state.objects.values()].map((item) => item.group),
    state.startMarker,
    state.goalMarker,
    state.roverGroup,
  ].filter(Boolean);
}

function updatePointer(event) {
  const bounds = renderer.domElement.getBoundingClientRect();
  pointer.x = ((event.clientX - bounds.left) / bounds.width) * 2 - 1;
  pointer.y = -((event.clientY - bounds.top) / bounds.height) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);
}

function hitTerrain(event) {
  updatePointer(event);
  const intersections = raycaster.intersectObject(state.terrain, false);
  return intersections.length ? intersections[0] : null;
}

function draggableCell(group) {
  if (group.userData.role === "map-object") {
    return state.objects.get(group.userData.id)?.cell ?? null;
  }
  if (group.userData.role === "start") return state.start;
  if (group.userData.role === "goal") return state.goal;
  if (group.userData.role === "rover") return state.rover;
  return null;
}

function canDrop(group, cell) {
  if (group.userData.role === "map-object") {
    return ![state.start, state.goal, state.rover].some((point) => sameCell(point, cell))
      && !objectsAt(cell).some((item) => item.id !== group.userData.id);
  }
  if (group.userData.role === "start") {
    return !isBlocked(cell) && !sameCell(cell, state.goal);
  }
  if (group.userData.role === "goal") {
    return !isBlocked(cell) && !sameCell(cell, state.rover);
  }
  if (group.userData.role === "rover") return !isBlocked(cell) && !sameCell(cell, state.goal);
  return false;
}

async function dropGroup(group, cell) {
  if (!canDrop(group, cell)) {
    showToast("No se puede solapar con un obstáculo, el rover o el destino.");
    if (group.userData.role === "map-object") {
      const entity = state.objects.get(group.userData.id);
      if (entity) updateEntityPosition(entity);
    } else if (group.userData.role === "start") updateMarkerPosition(group, state.start);
    else if (group.userData.role === "goal") updateMarkerPosition(group, state.goal);
    else updateRoverTarget();
    return;
  }

  if (group.userData.role === "map-object") {
    const entity = state.objects.get(group.userData.id);
    if (entity) {
      const data = await apiRequest(`/api/objects/${encodeURIComponent(entity.id)}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(cell),
      });
      syncFromPython(data);
      showToast(`${kindLabel(entity.kind)} movido a F${cell.row + 1} C${cell.col + 1}`);
    }
  } else if (group.userData.role === "start") {
    const data = await apiRequest("/api/markers/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(cell),
    });
    syncFromPython(data);
  } else if (group.userData.role === "goal") {
    const data = await apiRequest("/api/markers/goal", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(cell),
    });
    syncFromPython(data);
  } else if (group.userData.role === "rover") {
    showToast("El rover se mueve mediante la misión; mueve el inicio antes de empezar.");
    updateRoverTarget();
  }
}

async function placeAt(cell) {
  if (!cell) return;
  if (state.tool === "start") {
    if (isBlocked(cell) || sameCell(cell, state.goal)) {
      showToast("El inicio debe estar en una celda libre, distinta del destino.");
      return;
    }
    const data = await postAndSync("/api/markers/start", cell);
    if (data) showToast(`Inicio establecido · F${cell.row + 1} C${cell.col + 1}`);
    return;
  }
  if (state.tool === "goal") {
    if (isBlocked(cell) || sameCell(cell, state.rover)) {
      showToast("El destino debe estar libre y no puede estar bajo el rover.");
      return;
    }
    const data = await postAndSync("/api/markers/goal", cell);
    if (data) showToast(`Destino establecido · F${cell.row + 1} C${cell.col + 1}`);
    return;
  }
  if (state.tool === "erase") {
    const found = objectsAt(cell);
    for (const item of found) {
      const data = await apiRequest(`/api/objects/${encodeURIComponent(item.id)}`, {
        method: "DELETE",
      });
      syncFromPython(data);
    }
    if (found.length) {
      showToast(`${found.length} elemento(s) retirado(s).`);
    }
    return;
  }
  if (["rock", "crater", "gas", "rough", "cliff", "science"].includes(state.tool)) {
    await addObject(state.tool, cell);
  }
}

function onPointerDown(event) {
  if (event.button !== 0) return;
  updatePointer(event);
  const intersections = raycaster.intersectObjects(pickables(), true);
  const group = intersections.map(pickGroup).find(Boolean);
  state.dragStartPointer = { x: event.clientX, y: event.clientY };
  state.dragMoved = false;
  if (group) {
    state.dragged = {
      group,
      originalCell: { ...draggableCell(group) },
    };
    orbit.enabled = false;
    renderer.domElement.setPointerCapture(event.pointerId);
  }
}

function onPointerMove(event) {
  if (!state.dragged) return;
  const deltaX = event.clientX - state.dragStartPointer.x;
  const deltaY = event.clientY - state.dragStartPointer.y;
  if (Math.hypot(deltaX, deltaY) > 3) state.dragMoved = true;
  const intersection = hitTerrain(event);
  const cell = intersection && cellFromIntersection(intersection);
  if (!cell) return;
  const group = state.dragged.group;
  if (group.userData.role === "map-object") {
    const entity = state.objects.get(group.userData.id);
    if (entity) {
      entity.cell = cell;
      updateEntityPosition(entity);
    }
  } else if (group.userData.role === "start") {
    updateMarkerPosition(group, cell);
  } else if (group.userData.role === "goal") {
    updateMarkerPosition(group, cell);
  } else if (group.userData.role === "rover") {
    group.position.copy(cellWorldPosition(cell, 0.04));
  }
  el("coordX").textContent = String(cell.col + 1).padStart(2, "0");
  el("coordZ").textContent = String(cell.row + 1).padStart(2, "0");
}

function onPointerUp(event) {
  if (state.dragged) {
    const intersection = hitTerrain(event);
    const cell = intersection && cellFromIntersection(intersection);
    const { group, originalCell } = state.dragged;
    const finalCell = state.dragMoved && cell ? cell : originalCell;
    state.dragged = null;
    orbit.enabled = true;
    if (state.dragMoved) {
      if (group.userData.role === "map-object") {
        const entity = state.objects.get(group.userData.id);
        if (entity) entity.cell = originalCell;
      }
      dropGroup(group, finalCell).catch((error) => {
        setStatus(error.message, "error");
        showToast(error.message);
        const entity = group.userData.role === "map-object"
          ? state.objects.get(group.userData.id) : null;
        if (entity) updateEntityPosition(entity);
      });
    } else if (group.userData.role === "map-object" && state.tool === "erase") {
      apiRequest(`/api/objects/${encodeURIComponent(group.userData.id)}`, { method: "DELETE" })
        .then(syncFromPython)
        .catch((error) => showToast(error.message));
    }
    state.dragStartPointer = null;
    return;
  }

  if (!state.dragStartPointer) return;
  const deltaX = event.clientX - state.dragStartPointer.x;
  const deltaY = event.clientY - state.dragStartPointer.y;
  state.dragStartPointer = null;
  if (Math.hypot(deltaX, deltaY) > 4) return;
  const intersection = hitTerrain(event);
  if (intersection) {
    placeAt(cellFromIntersection(intersection)).catch((error) => {
      setStatus(error.message, "error");
      showToast(error.message);
    });
  }
}

function scheduleConfiguration() {
  clearTimeout(state.configTimer);
  state.configTimer = setTimeout(async () => {
    if (!state.pythonState) return;
    const config = {
      risk_weight: Number(el("riskWeight").value),
      gas_concentration: Number(el("gasLevel").value) / 100,
      science_priority: Number(el("sciencePriority").value) / 100,
      strategy: el("strategy").value,
    };
    if (state.steps === 0) config.battery = Number(el("battery").value);
    const data = await postAndSync("/api/config", config);
    if (data) setStatus("Parámetros modificados · rutas y decisión actualizadas.");
  }, 220);
}

async function injectEvent(eventName) {
  if (state.playing) stopMission("Misión pausada para registrar un evento.");
  const data = await postAndSync("/api/events", { event: eventName });
  if (data) showToast(data.events.at(-1)?.message || "Evento inyectado.");
}

async function injectFault() {
  if (state.playing) stopMission("Misión pausada para comprobar sensores.");
  const data = await postAndSync("/api/faults", {
    fault: el("faultSelect").value,
    enabled: true,
  });
  if (data) showToast(`Fallo de sensor activado: ${el("faultSelect").selectedOptions[0].text}.`);
}

async function restoreSensors() {
  if (state.playing) stopMission("Misión pausada para restaurar sensores.");
  const activeFaults = state.pythonState?.faults || [];
  for (const fault of activeFaults) {
    const data = await apiRequest("/api/faults", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ fault, enabled: false }),
    });
    syncFromPython(data);
  }
  showToast("Sensores restaurados.");
}

async function clearMapObjects() {
  if (state.playing) stopMission("Misión pausada para editar el mapa.");
  const data = await postAndSync("/api/map/clear", {});
  if (data) showToast("Mapa despejado. Inicio y destino se conservaron.");
}

async function runExperiment() {
  const button = el("runExperiment");
  button.disabled = true;
  button.textContent = "Ejecutando 20 semillas…";
  el("experimentStatus").className = "experiment-status";
  el("experimentStatus").textContent = "Comparando 20 semillas; puede tardar alrededor de 1–2 minutos.";
  try {
    const started = await apiRequest("/api/experiments", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ seeds: 20 }),
    });
    const poll = async () => {
      const current = await apiRequest(`/api/experiments/${started.id}`);
      if (current.status === "running") {
        state.experimentTimer = setTimeout(poll, 800);
        return;
      }
      renderExperimentResults(current.result);
      el("experimentStatus").className = "experiment-status success";
      el("experimentStatus").textContent = "Experimento finalizado · 20 semillas idénticas entre políticas.";
      button.disabled = false;
      button.textContent = "Comparar 20 semillas y ablaciones";
    };
    await poll();
  } catch (error) {
    el("experimentStatus").className = "experiment-status error";
    el("experimentStatus").textContent = error.message;
    button.disabled = false;
    button.textContent = "Reintentar comparación";
  }
}

function renderExperimentResults(result) {
  const names = {
    hibrido: "Sistema híbrido",
    sin_rna: "Sin RNA",
    sin_difuso: "Sin lógica difusa",
    sin_ag: "Sin AG",
  };
  el("experimentResults").innerHTML = `<table class="experiment-table"><thead><tr><th>POLÍTICA</th><th>ÉXITO</th><th>DISTANCIA MEDIA ± DE</th><th>BATERÍA MEDIA</th><th>CIENCIA MEDIA</th><th>VETOS MEDIOS</th></tr></thead><tbody>${Object.entries(result.policies).map(([key, value]) => `<tr><td>${names[key] || key}</td><td>${value.success_rate_pct}%</td><td>${value.mean_distance_m} ± ${value.std_distance_m} m</td><td>${value.mean_battery_remaining_pct}%</td><td>${value.mean_science_collected}</td><td>${value.mean_safety_violations}</td></tr>`).join("")}</tbody></table><div class="diagnostic-note">${result.note} · ${result.seed_count} semillas.</div>`;
}

function initUI() {
  document.querySelectorAll(".tool").forEach((button) => {
    button.addEventListener("click", () => {
      if (!button.disabled) syncTool(button.dataset.tool);
    });
  });
  el("planet").addEventListener("change", syncPlanet);
  el("strategy").addEventListener("change", () => {
    state.strategy = el("strategy").value;
    el("routeModeLabel").textContent = state.strategy === "safe" ? "MENOR RIESGO" : "DISTANCIA";
    scheduleConfiguration();
  });
  el("riskWeight").addEventListener("input", () => {
    el("riskValue").textContent = Number(el("riskWeight").value).toFixed(2);
    scheduleConfiguration();
  });
  el("gasLevel").addEventListener("input", () => {
    el("gasValue").textContent = `${el("gasLevel").value}%`;
    scheduleConfiguration();
  });
  el("sciencePriority").addEventListener("input", () => {
    el("sciencePriorityValue").textContent = `${el("sciencePriority").value}%`;
    scheduleConfiguration();
  });
  el("battery").addEventListener("input", () => {
    if (state.steps === 0) {
      state.energy = Number(el("battery").value);
      updateStats();
      scheduleConfiguration();
    } else {
      showToast("Reinicia la misión para cambiar la batería inicial.");
    }
  });
  el("speed").addEventListener("input", () => {
    el("speedValue").textContent = `${el("speed").value} ms`;
    if (state.playing) {
      clearInterval(state.timer);
      state.timer = setInterval(() => { advanceOneStep(); }, Number(el("speed").value));
    }
  });
  el("play").addEventListener("click", togglePlay);
  el("step").addEventListener("click", () => {
    if (state.playing) {
      showToast("Pausa la misión antes de avanzar manualmente.");
      return;
    }
    advanceOneStep();
  });
  el("reset").addEventListener("click", resetMission);
  el("clear").addEventListener("click", clearMapObjects);
  document.querySelectorAll(".event-button[data-event]").forEach((button) => {
    button.addEventListener("click", () => injectEvent(button.dataset.event));
  });
  el("injectFault").addEventListener("click", injectFault);
  el("clearFault").addEventListener("click", restoreSensors);
  el("runExperiment").addEventListener("click", runExperiment);
  el("viewReset").addEventListener("click", () => {
    camera.position.set(17, 20, 23);
    orbit.target.set(0, 0, 0);
    orbit.update();
  });
}

function onResize() {
  const width = sceneElement.clientWidth;
  const height = sceneElement.clientHeight;
  if (!width || !height) return;
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  renderer.setSize(width, height, false);
}

function animate() {
  requestAnimationFrame(animate);
  orbit.update();
  updateCosmos(performance.now());
  if (state.roverFrom && state.roverTo && state.roverGroup) {
    const amount = Math.min(1, (performance.now() - state.roverMoveStarted) / 360);
    state.roverGroup.position.lerpVectors(state.roverFrom, state.roverTo, amount);
    if (amount >= 1) state.roverFrom = null;
  }
  renderer.render(scene, camera);
}

function start() {
  createTerrain();
  createCosmos();
  state.startMarker = makeMarker("start");
  state.goalMarker = makeMarker("goal");
  updateMarkerPosition(state.startMarker, state.start);
  updateMarkerPosition(state.goalMarker, state.goal);
  state.roverGroup = createRover();
  updateRoverTarget();
  initUI();
  el("routeModeLabel").textContent = state.strategy === "safe" ? "MENOR RIESGO" : "DISTANCIA";
  setStatus("Conectando motor Python…");
  onResize();
  new ResizeObserver(onResize).observe(sceneElement);
  renderer.domElement.addEventListener("pointerdown", onPointerDown);
  renderer.domElement.addEventListener("pointermove", onPointerMove);
  renderer.domElement.addEventListener("pointerup", onPointerUp);
  renderer.domElement.addEventListener("pointercancel", onPointerUp);
  renderer.domElement.addEventListener("contextmenu", (event) => event.preventDefault());
  animate();
  bootstrapEngine();
}

start();
