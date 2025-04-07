import bpy
import bmesh

# 获取当前选中物体（必须是网格）
obj = bpy.context.scene.objects.get('Cube') 
if not obj or obj.type != 'MESH':
    raise Exception("请选中一个网格物体")

# 要分离的面片索引（示例：分离第0和第2个面）
face_indices = [0, 3]

# 进入编辑模式
bpy.ops.object.mode_set(mode='EDIT')

# 初始化 BMesh
bm = bmesh.from_edit_mesh(obj.data)
bm.faces.ensure_lookup_table()  # 确保面片索引可用

# 选中指定面片

for face in bm.faces:
    face.select = False  # 先取消所有选择
for idx in face_indices:
    if idx < len(bm.faces):
        bm.faces[idx].select = True  # 选中目标面片
        # 分离选中面片
        bpy.ops.mesh.separate(type='SELECTED')  # 分离操作
        # bm.faces[idx].select = False 
        
# 退出编辑模式

# 更新网格数据
bmesh.update_edit_mesh(obj.data)

bpy.ops.object.mode_set(mode='OBJECT')
