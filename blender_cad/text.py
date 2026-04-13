import bpy
from typing import List, Optional, Tuple, Union
from typing_extensions import override

from .material import mat
from .curve import BaseCurve, Curve
from .location import Location

class t:
    """
    Text fragment class for character-level styling.
    Builds a tree of text segments, propagating styles like bold, italic, and materials.
    Supports concatenation with python strings using operator overloading.
    """
    def __init__(
        self, 
        text: Union[str, 't', List['t']] = "", 
        mat: Optional[mat.Layer] = None, 
        bold: Optional[bool] = None, 
        italic: Optional[bool] = None
    ):
        self.children: List[Union[str, 't']] = []
        if isinstance(text, str):
            self.children.append(text)
        elif isinstance(text, t):
            self.children.append(text)
        elif isinstance(text, list):
            self.children.extend(text)
        
        self.mat = mat
        self.bold = bold
        self.italic = italic

    def __add__(self, other: Union[str, 't']) -> 't':
        """Allows: t('hello') + 'world'"""
        res = t()
        res.children = [self, other if isinstance(other, (t, str)) else str(other)]
        return res
        
    def __radd__(self, other: Union[str, 't']) -> 't':
        """Allows: 'hello' + t('world')"""
        res = t()
        res.children = [other if isinstance(other, (t, str)) else str(other), self]
        return res
        
    def __iadd__(self, other: Union[str, 't']) -> 't':
        """Allows: my_t += '!!!'"""
        self.children.append(other if isinstance(other, (t, str)) else str(other))
        return self

    @classmethod
    def b(cls, text: Union[str, 't'], **kwargs) -> 't':
        """Shortcut for bold text."""
        return cls(text, bold=True, **kwargs)
        
    @classmethod
    def i(cls, text: Union[str, 't'], **kwargs) -> 't':
        """Shortcut for italic text."""
        return cls(text, italic=True, **kwargs)
        
    @property
    def plain(self) -> str:
        """Returns the raw unformatted string from the tree."""
        return "".join(
            child if isinstance(child, str) else child.plain 
            for child in self.children
        )
        
    def flatten(self, parent_style: Optional[dict] = None) -> List[Tuple[str, dict]]:
        """
        Recursively flattens the tree into a list of (character, style_dict) tuples.
        Child styles override parent styles.
        """
        if parent_style is None:
            parent_style = {}
            
        current_style = parent_style.copy()
        if self.mat is not None: current_style['mat'] = self.mat
        if self.bold is not None: current_style['bold'] = self.bold
        if self.italic is not None: current_style['italic'] = self.italic
        
        result = []
        for child in self.children:
            if isinstance(child, str):
                for char in child:
                    result.append((char, current_style))
            else:
                result.extend(child.flatten(current_style))
        return result


class Text(BaseCurve):
    """
    An object representing a 3D Text curve in Blender.
    Supports character-level formatting, standard geometry generation, and curve pathing.
    """
    def __init__(self, text: Union[str, t] = "", size: float = 1.0, loc: Location = Location(), obj: Optional[bpy.types.Object] = None):
        super().__init__(obj)
        self.loc = loc
        self._text = t()
        
        # Initial overall object properties
        self.obj.data.size = size
        self.obj.data.align_x = 'LEFT'
        self.obj.data.align_y = 'BOTTOM_BASELINE'
        
        # Triggers the property setter and builds the text
        self.text = text 

    @override
    def _create_empty_object(self):
        """Creates a new Blender FONT object."""
        txt_data = bpy.data.curves.new(name="TextData", type='FONT')
        obj = bpy.data.objects.new("Text", txt_data)
        return obj

    @override
    def copy(self) -> 'Text':
        """Creates a copy of the Text and its underlying Blender object."""
        if not self.is_valid:
            raise RuntimeError("Object is removed")
        
        # Copy object and data
        new_obj = self.obj.copy()
        new_obj.data = self.obj.data.copy()
        new_text = Text(self.text, size=self.size, loc=self.loc, obj=new_obj)
        return new_text

    @property
    def text(self) -> t:
        """Returns the root text fragment tree."""
        return self._text
        
    @text.setter
    def text(self, value: Union[str, t]):
        """Sets the text and triggers a geometry rebuild."""
        if isinstance(value, str):
            self._text = t(value)
        elif isinstance(value, t):
            self._text = value
        self._rebuild()

    def _rebuild(self):
        """Rebuilds the Blender text object based on the current text tree state."""
        if not self.obj or not self.obj.data:
            return
            
        txt_data = self.obj.data
        flattened = self._text.flatten()
        
        # Update raw body text
        txt_data.body = self._text.plain
            
        # Apply character formatting matching the body string length
        # Blender's body_format array has the same length as the string in `body`
        for i, (char, style) in enumerate(flattened):
            if i >= len(txt_data.body_format):
                break
            fmt = txt_data.body_format[i]
            
            # Apply Material
            mat = style.get('mat')
            if mat:
                fmt.material_index = self._get_or_create_material_index(mat)
                
            # Apply Typography
            fmt.use_bold = style.get('bold', False)
            fmt.use_italic = style.get('italic', False)

    @property
    def font_size(self) -> float:
        return self.obj.data.size
        
    @font_size.setter
    def font_size(self, value: float):
        self.obj.data.size = value

    @property
    def spacing_character(self) -> float:
        return self.obj.data.space_character
        
    @spacing_character.setter
    def spacing_character(self, value: float):
        self.obj.data.space_character = value

    def align(self, x: str = 'CENTER', y: str = 'CENTER') -> 'Text':
        """Sets text alignment. X: LEFT, CENTER, RIGHT. Y: TOP, CENTER, BOTTOM."""
        self.obj.data.align_x = x
        self.obj.data.align_y = y
        return self

    def load_fonts(self, regular: str, bold: Optional[str] = None, italic: Optional[str] = None, bold_italic: Optional[str] = None) -> 'Text':
        """Loads TTF/OTF files and assigns them to the object font slots."""
        def get_font(path):
            return bpy.data.fonts.load(path) if path else None
            
        self.obj.data.font = get_font(regular)
        if bold: self.obj.data.font_bold = get_font(bold)
        if italic: self.obj.data.font_italic = get_font(italic)
        if bold_italic: self.obj.data.font_bold_italic = get_font(bold_italic)
        return self

    def put_on_curve(self, curve: Curve) -> 'Text':
        """
        Deforms the text along a path.
        """
        self.loc = Location()
        mod: bpy.types.CurveModifier = self.obj.modifiers.new(name="CurvePath", type='CURVE')
        mod.object = curve.obj

        # Store dependency so .part knows what to bring onto the scene
        self._dependencies = [curve]
        return self
    