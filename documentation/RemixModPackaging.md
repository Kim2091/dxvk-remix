# Packaging a Remix mod

How to capture a scene for the Remix Toolkit, and how to package a Remix mod so that players' mod managers decide what happens when it and another mod replace the same asset. How the runtime uses the result: [RemixModOrder.md](RemixModOrder.md). What players do: [RemixModsForPlayers.md](RemixModsForPlayers.md).

## Capturing a scene (New Vegas RTX)

A capture is a USD stage of what the game draws: every mesh, material and light under the name a mod replaces it by, with their textures and geometry. The Remix Toolkit opens it, and a mod made from it applies in game.

1. **Use matching files.** The Remix runtime, the bridge, `remix-comp` and `fnv_remix_nvse.dll` must come from one New Vegas RTX release, and `rtx.conf` must have `rtx.scene.assetKeys = True` (New Vegas RTX ships it). With an older bridge, `remix-comp.log` says `the bridge client does NOT forward AssetKeyEXT`, and meshes drawn through the Remix API get no stable name.
2. **Go to the place** you want to work on, for example Goodsprings, and wait until it has finished loading.
3. **Open the Remix menu** (Alt+X by default), tab **Enhancements**, and switch off **Enable Enhancements**. A scene cannot be captured while replacements are on: the capture panel then says "Disable enhanced assets to enable capturing."
4. **Limit the capture.** An exterior holds the loaded cells, the distant LOD and objects kept while out of view, often tens of thousands of instances: such a capture takes minutes and gigabytes. Set **Max instance distance** to the radius you need, in Remix's world units. An exterior cell is about 58.5 m wide: 4096 units when remix-comp hands Remix the game's own units, 5851 when it hands over centimetres (`[RemixApi] WorldUnitScale`). The sky is always captured. Grass and other instanced draws keep the placements nearest the camera (**Max placements per instanced draw**).
5. **Capture Scene.** Give it a name and press the button. The stage and its `textures`, `meshes`, `materials` and `lights` folders are written to `rtx-remix\captures\`. The scene is taken from one frame; the files are written over the next seconds, while the progress bar shows.
6. **Switch Enable Enhancements back on.**
7. **Open the capture in the Toolkit** (a new project on the game's `rtx-remix` folder, with the capture as its source). Replace materials (`/RootNode/Looks/mat_<HASH>`), meshes (`/RootNode/meshes/mesh_<HASH>`) and lights (`/RootNode/lights/light_<HASH>`) as for any game, then package the mod as below.
8. **Check it in game.** Install the mod in `rtx-remix\mods\`, start the game (or press **Reload Enhancements**) and look at the same place. **Enhancements > Mod Conflicts** shows how many materials, meshes and lights the mod replaces.

### What the names are

The hash in each name is the same in every session and on every PC, so a mod made from one capture applies for every player.

| Asset | Name | Hash |
|-------|------|------|
| Material | `mat_<HASH>` | The game texture's hash, the one the Remix menu's texture picker shows and a D3D9 draw of the texture uses. Several draws of one texture share one material. |
| Mesh | `mesh_<HASH>` | The geometry's hash: the runtime's default geometry asset hash rule (`positions,indices,geometrydescriptor`) over the mesh as remix-comp sends it, in the object space of the game's mesh. Distant LOD blocks are hashed before they sink into the terrain. A mod that edits the game's mesh changes the hash. |
| Light | `light_<HASH>` | A light placed in the world (a lamp reference, or a light inside a placed object's model): its reference and base form, the reference's position to the unit and the light's place in the model. The form IDs are taken without the load order index, so the hash does not change with the load order. The Pip-Boy light has one fixed hash. |

Not replaceable by name:

- Lights that belong to effects (muzzle flashes, explosions, projectiles, spells) are named by their position, which changes every frame.
- Particles are rebuilt every frame and have no stable mesh name. Replace their material instead.
- A material whose texture remix-comp uploads itself instead of importing the game's texture (`[RemixApi] ImportD3D9Textures=0`, or a texture format the import refuses) is named by its content hash, which changes between releases.
- Land and water are captured, but replacing their meshes or materials is untested.

### Mods made before these names

- Material replacements named by the texture picker's hash keep working: that has always been the material's name.
- Mesh replacements made against a D3D9 draw's mesh hash do not apply to objects drawn through the Remix API (most of the world), because those meshes are hashed from the vertices remix-comp sends, not from the game's vertex buffer. Capture again and move the replacement to the new `mesh_<HASH>`.
- Earlier captures named API meshes by per-session handles, API materials by hashes of their settings and lights by their camera-relative position, so meshes and lights made from them never matched. Capture again.

## Steps

1. **Export from the Remix Toolkit** (Mod Packaging). Make sure the mods yours builds on are its dependencies: the Toolkit writes them into `lightspeed_mod_dependencies` in `mod.usda`, and the runtime ranks your mod above them, so your changes win where you override theirs. A dependency may name the other mod's folder, a path to it or to its `mod.usda`, or its `lightspeed_mod_name`.
2. **Lay out the folder** the way it goes into the game folder:
   ```
   rtx-remix\
     mods\
       MyMod\
         mod.usda
         (sublayers, textures, meshes...)
   ```
   The folder name (`MyMod`) is your mod's identity. Keep it the same in every version: ownership files and other mods' dependency lists name it.
3. **Write the ownership files** with `remix_mod_owners` from that folder's parent:
   ```
   remix_mod_owners.exe rtx-remix\mods\MyMod
   ```
   It reads `mod.usda` with its sublayers, finds every material, mesh and light the mod replaces (`mat_<HASH>`, `mesh_<HASH>`, `light_<HASH>` under `/RootNode`), and writes one file per asset to `rtx-remix\mods\owners\`, each holding `MyMod`. Run it again after every export. Add `--prune` to delete the files of assets you no longer replace.
4. **Archive** the `rtx-remix` folder, `owners` included:
   ```
   rtx-remix\mods\MyMod\...
   rtx-remix\mods\owners\mat_0123456789ABCDEF
   rtx-remix\mods\owners\mesh_...
   ```
   The folder is `owners`, with no leading dot: Mod Organizer 2's Root Builder does not copy names that start with a dot.
   This layout installs by hand and with Vortex. For Mod Organizer 2 with Root Builder, offer a second archive with the same content under `Root\` (`Root\rtx-remix\mods\...`), or tell players to move it there.

## Notes

- Run the tool in your packaging folder, not in a game folder with other Remix mods installed: it overwrites the ownership files it finds, other mods' included. In a game folder, `--check` lists which of your assets other installed mods also replace without changing anything.
- Ship only the ownership files the tool writes for your mod. A file for an asset your mod does not replace is ignored, and one naming another mod makes yours lose that asset.
- A material's hash is the game texture's hash, the one the Remix menu's texture picker shows. For games drawn through the Remix API it is matched as well, as long as the game imports its own textures. Captures name such materials by it.
- `remix_mod_owners.exe` runs on the CPU and needs the USD DLLs and the `usd` plugin folder beside it, as `tools\remix_mod_owners\build.bat` lays them out in `bin\`.
