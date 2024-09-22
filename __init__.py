from .nodes.load_dust3r_node import LoadDust3rNode
from .nodes.run_dust3r_node import RunDust3rNode
from .nodes.render_dust3r_node import RenderDust3rSimpleNode


NODE_CLASS_MAPPINGS = {
    "LoadDust3r": LoadDust3rNode,
    "RunDust3r": RunDust3rNode,
    "RenderDust3rSimple": RenderDust3rSimpleNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadDust3r": "Load Dust3r",
    "RunDust3r": "Run Dust3r",
    "RenderDust3rSimple": "Render Dust3r"
}
