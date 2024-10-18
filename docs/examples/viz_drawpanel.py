"""
=========
DrawPanel
=========

This example shows how to use the DrawPanel UI. We will demonstrate how to
create Various shapes and transform them.

First, some imports.
"""

import fury

##############################################################################
# First we need to fetch some icons that are needed for DrawPanel.

fury.data.fetch_viz_new_icons()

#########################################################################
# We then create a DrawPanel Object.

drawing_canvas = fury.ui.DrawPanel(size=(560, 560), position=(40, 10))

###############################################################################
# Show Manager
# ============
#
# Now we add DrawPanel to the scene.

current_size = (650, 650)
showm = fury.window.ShowManager(size=current_size, title="DrawPanel UI Example")

showm.scene.add(drawing_canvas)

interactive = False

if interactive:
    showm.start()
else:
    # If the UI isn't interactive, then adding a circle to the canvas
    drawing_canvas.current_mode = "circle"
    drawing_canvas.draw_shape(shape_type="circle", current_position=(275, 275))
    drawing_canvas.shape_list[-1].resize((50, 50))

    fury.window.record(
        scene=showm.scene, size=current_size, out_path="viz_drawpanel.png"
    )
