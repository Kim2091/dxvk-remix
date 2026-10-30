# Remix mod order

How the runtime decides which Remix mod wins when several replace the same game asset. For installing mods see [RemixModsForPlayers.md](RemixModsForPlayers.md); for packaging them see [RemixModPackaging.md](RemixModPackaging.md).

## Mods

Every folder under `rtx-remix/mods/` that holds a `mod.usda`, `mod.usd` or `mod.usdc` is one mod (`ModManager::enumerateAllMods`). Its folder name is its identity: ownership files and dependency lists name a mod by it. A folder without a mod file, such as `owners`, is not a mod.

Each mod replaces game assets by hash, named by its prims under `/RootNode`:

| Kind | Prims | Ownership file |
|------|-------|----------------|
| Material | children of `/RootNode/Looks` named `mat_<HASH>` | `mat_<HASH>` |
| Mesh | children of `/RootNode/meshes` named `mesh_<HASH>` | `mesh_<HASH>` |
| Light | children of `/RootNode/lights` named `light_<HASH>` (or the legacy `sphereLight_<HASH>`) | `light_<HASH>` |

`<HASH>` is the hash as 16 upper-case hex digits. `src/lssusd/mod_asset_keys.h` reads these names from a composed stage, so the runtime and the `remix_mod_owners` tool agree on what a mod replaces. Two mods clash on an asset when both replace it.

## Precedence

Mods are kept in one precedence order, lowest first (`ModManager::orderMods`):

1. The mods are sorted by folder name, ignoring case. A later name has the higher precedence.
2. Each mod is moved after every mod its `lightspeed_mod_dependencies` names, so a mod beats the mods it builds on. A dependency is matched by folder name, by a path to the folder or its mod file, or by `lightspeed_mod_name`. Among the mods whose dependencies are placed, the earliest name goes next. A dependency cycle is broken by name and logged.

The winner of a clash is decided by the first of these rules that applies (`ModManager::resolveClashes`):

1. **Ownership file.** `rtx-remix/mods/owners/<kind>_<HASH>` exists and its first line names one of the mods replacing the asset. That mod wins.
2. **Precedence order.** Otherwise the clashing mod with the highest precedence wins. The conflicts view and the log name the reason: `dependency` when the winner depends on every other clashing mod, `name` when its folder name is the latest, `load order` when the order came from both.

An ownership file that names a mod which is not installed, or which does not replace that asset, is stale. It is ignored, the precedence order decides, and the log and the conflicts view show the stale name.

Every lookup (`AssetReplacer::getReplacementMaterial`, `getReplacementsForMesh`, `getReplacementsForLight`) asks the clash winner first and then walks the mods from the highest precedence down. The texture search paths (`ModManager::publishSearchPaths`, walked backwards by `AssetDataManager::findAsset`) use the same order, so the mod that wins its assets also wins its RTX IO packages. Ownership files do not apply to textures: textures are files, not hashed assets.

## Why ownership files

Mod managers resolve file conflicts with an order the player controls: MO2's left-pane priority (Root Builder deploys in that order), Vortex's load-after rules, or the last archive extracted by hand. Two mods replacing the same asset ship the same ownership file path, each naming itself, so whichever file the mod manager lets through is the player's choice, and the runtime follows it. No mod manager has to know about Remix.

## When it is resolved

- The precedence order is set once, when the mods are enumerated at device creation. Each mod's metadata (`UsdMod::readModInfo`) is read from the layer header only.
- The replaced assets of a mod are collected after each load or hot reload (`UsdMod::Impl::rebuildWorkerEntry`) and handed to the manager when the rebuild is applied. Every hand-over re-resolves the clashes. Only the assets that clash have their ownership files read, and each file is read once until **Reload Enhancements** is pressed.
- When a clash gets a new winner, the next `AssetReplacer::checkForChanges` reports a mod change, so the scene drops what it resolved against the previous winner. With two or more mods loading in turn at startup this happens once, after the second clashing mod loads.

## Logs

`[ModOrder]` lines in `remix-dxvk.log`:

```
[ModOrder] 2 Remix mod(s), from lowest to highest precedence:
[ModOrder]   1. TestA ("Mod order test A" 1.0.0)
[ModOrder]   2. TestB ("Mod order test B" 1.0.0)
[ModOrder] 1 asset(s) replaced by more than one mod:
[ModOrder]   mat_C5D7755CE723D80D: TestA wins over TestB (ownership file)
```

At most 32 clash lines are written per change. The Remix menu's **Enhancements > Mod Conflicts** lists all of them, with the replaced-asset counts per mod.

## Limits

- A material replacement keyed by an API material's content hash and one keyed by its game texture hash replace the same surface without clashing; the content hash is looked up first ([CHANGELOG](../docs/CHANGELOG.md), 2026-10-20).
- With `rtx.scene.assetKeys`, an API mesh or a D3D9 light that has a host asset key is replaced by that key, and a capture names it by the key too ([RemixModPackaging.md](RemixModPackaging.md#what-the-names-are)).
- Secret replacement variants (`mesh hash + variant id`) do not take part in clash resolution.
- Changing an ownership file takes effect at the next **Reload Enhancements** or restart.
