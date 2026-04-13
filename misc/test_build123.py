from build123d import *
from ocp_vscode import show

# КАК ЗАПУСТИТЬ:
# 1) Выбери интерпретатор с либой build123d и ocp_vscode
# 2) Запусти ocp viewer: python -m ocp_vscode
# 3) Запусти скрипт. В viewer появится результат.

def Test():
    # 1. Создаем объекты
    with BuildPart() as p:
        with Locations((0, 0, 0)):
            Box(10, 20, 30)
        # Выбираем верхнюю грань
        top_face = p.faces().sort_by(Axis.Z)[-1]
        with Locations(top_face.at(0.0, 0.0)):
            Sphere(1)
        # Выбираем самое длинное ребро на этой грани
        long_edge = top_face.edges().sort_by(SortBy.LENGTH)[-1]

    with BuildPart() as cyl_p:
        with Locations((30, 0, 0)):
            Cylinder(radius=8, height=30)
        # Выбираем именно боковую поверхность (цилиндрическую)
        top_face = cyl_p.faces().sort_by(-Axis.Z)[0]
        side_surface = cyl_p.faces().sort_by(Axis.X)[0]
        with Locations(top_face.at(0.0, 0.0)):
            Sphere(0.5)
        with Locations(top_face.at(0.5, 0.5)):
            Sphere(0.5)
        with Locations(top_face.at(1.0, 1.0)):
            Sphere(0.5)

        with Locations(side_surface.at(0.0, 0.0)):
            Sphere(1)
        with Locations(side_surface.at(0.7, 0.7)):
            Sphere(1)
        with Locations(side_surface.at(1.0, 1.0)):
            Sphere(1)

        # Выбираем верхний ободок (edge)
        top_rim = cyl_p.edges().sort_by(Axis.Z)[-1]

    with BuildPart() as sph_p:
        with Locations((60, 0, 0)):
            Sphere(12)
        sph_face = sph_p.faces()[0]
        with Locations(sph_face.at(0.0, 0.0)):
            Sphere(1)
        with Locations(sph_face.at(0.4, 0.4)):
            Sphere(1)
        with Locations(sph_face.at(1.0, 1.0)):
            Sphere(1)

    # Собираем всё, что хотим "подсветить"
    selections = [top_face, long_edge, side_surface, top_rim, sph_face]
    
    # Визуализация в расширении:
    # 'p' и остальные — это полупрозрачные детали
    # 'selections' — яркие выделенные элементы
    show(p, cyl_p, sph_p, selections, names=["Box", "Cyl", "Sphere", "Selected"])
    
    return p

if __name__ == "__main__":
    Test()
