"""PPT-style click-through curve guides on the Trench image."""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import numpy as np
from PySide6.QtCore import QObject, QEvent, QPointF, Qt, Signal
from PySide6.QtGui import QColor, QPainterPath, QPainterPathStroker, QPen, QTransform
from shiboken6 import isValid
from .trench_guides import normalize_guides, curve_samples, sweep_samples


def guide_sweep_path(guide):
    """Union of equal-radius disks moving along the sampled curve.

    One winding-filled, simplified stroke removes interior overlap edges;
    overlapping disks are painted once instead of accumulating opacity.
    """
    points=tuple(tuple(map(float,p)) for p in guide['points'])
    return QPainterPath(_sweep_path(points,float(guide['half_width_px'])))


@lru_cache(maxsize=16)
def _sweep_path(points,radius):
    samples=sweep_samples(dict(points=points))
    center=QPainterPath(QPointF(*samples[0]))
    for point in samples[1:]:center.lineTo(QPointF(*point))
    stroke=QPainterPathStroker()
    stroke.setWidth(2*radius)
    stroke.setCapStyle(Qt.PenCapStyle.RoundCap)
    stroke.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    path=stroke.createStroke(center)
    path.setFillRule(Qt.FillRule.WindingFill)
    # Qt flattens round arcs during the union. Work at higher precision so
    # round end caps also stay smooth when the SEM image is zoomed in.
    outline=QTransform.fromScale(1/16,1/16).map(QTransform.fromScale(16,16).map(path).simplified())
    # Qt's stroke union can leave tiny interior contours at overlapping joins.
    # A real boundary must be radius away from the centerline. Drop numerical
    # contours well inside the sweep, while retaining real enclosed holes.
    result=QPainterPath();result.setFillRule(outline.fillRule())
    a,delta=samples[:-1],np.diff(samples,axis=0)
    lengths=np.maximum(np.sum(delta*delta,axis=1),1e-20)
    for polygon in outline.toSubpathPolygons():
        point=np.array([polygon[0].x(),polygon[0].y()])
        along=np.clip(np.sum((point-a)*delta,axis=1)/lengths,0,1)
        distance=np.min(np.linalg.norm(point-(a+along[:,None]*delta),axis=1))
        if distance<radius-.25:continue
        result.addPolygon(polygon);result.closeSubpath()
    return result


class TrenchGuideEditor(QObject):
    changed = Signal()
    selection_changed = Signal()
    mode_changed = Signal(bool)
    message = Signal(str)

    def __init__(self, view):
        super().__init__(view)
        self.view = view
        self.guides = []
        self.selected = -1
        self.drawing = False
        self.side = 'contour'
        self.radius = 12.
        self.items = []
        self.drag = None
        self.draft = []
        self.view.viewport().setMouseTracking(True)
        view.installEventFilter(self)
        view.viewport().installEventFilter(self)
        view.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def clear_render(self):
        for item in self.items:
            self.view.scene().removeItem(item)
        self.items = []

    def restore(self, guides=()):
        self.clear_render()
        self.guides = list(normalize_guides(guides))
        self.selected = -1
        self.drag = None
        self.set_drawing(False)
        self.render()
        self.selection_changed.emit()

    def set_drawing(self, enabled):
        self.drawing = bool(enabled)
        self.drag = None
        self.draft = []
        if enabled:self.selected=-1
        self.view.set_select_region(False)
        self.view.setCursor(Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.OpenHandCursor)
        self.render()
        self.mode_changed.emit(self.drawing)
        self.selection_changed.emit()

    def select(self, index):
        self.selected = index if 0 <= index < len(self.guides) else -1
        self.render()
        self.selection_changed.emit()

    def set_radius(self, radius):
        self.radius = float(radius)
        if self.drawing:
            self.render(self._draft_guide())
        elif self.selected >= 0 and self.guides[self.selected]['half_width_px'] != radius:
            self.guides[self.selected]['half_width_px'] = float(radius)
            self.render()
            self.changed.emit()

    def remove_selected(self):
        if self.selected >= 0:
            self.guides.pop(self.selected)
            self.selected = min(self.selected,len(self.guides)-1)
            self.render()
            self.changed.emit()
            self.selection_changed.emit()

    def clear(self):
        self.restore()
        self.changed.emit()

    def _point(self, event):
        p = self.view.mapToScene(event.position().toPoint())
        rect = self.view.image_rect
        return np.array([np.clip(p.x(),rect.left(),max(rect.left(),rect.right()-1)),
                         np.clip(p.y(),rect.top(),max(rect.top(),rect.bottom()-1))],float)

    def _hit(self, event, band=False):
        pos = np.array([event.position().x(),event.position().y()])
        order = ([self.selected] if self.selected>=0 else []) + [i for i in reversed(range(len(self.guides))) if i!=self.selected]
        for i in order:
            guide = self.guides[i]
            endpoints = [np.array([q.x(),q.y()]) for p in guide['points']
                         for q in [self.view.mapFromScene(QPointF(*p))]]
            for end, p in enumerate(endpoints):
                if np.linalg.norm(pos-p)<=9:
                    return i,end
            samples=curve_samples(guide)
            mapped=[self.view.mapFromScene(QPointF(*p)) for p in samples]
            pixels=np.array([[p.x(),p.y()] for p in mapped],float)
            a,b=pixels[:-1],pixels[1:];delta=b-a
            along=np.clip(np.sum((pos-a)*delta,axis=1)/np.maximum(1,np.sum(delta*delta,axis=1)),0,1)
            distance=float(np.min(np.linalg.norm(pos-(a+along[:,None]*delta),axis=1)))
            if distance<=max(8,guide['half_width_px']*self.view.transform().m11() if band else 8):
                return i,None
        return -1,None

    def render(self, preview=None):
        self.clear_render()
        guides=self.guides + ([preview] if preview else [])
        for i,guide in enumerate(guides):
            color=QColor('#79dfb6')
            if len(guide['points'])>=2:
                path=guide_sweep_path(guide)
                brush=QColor(color);brush.setAlpha(42 if i!=self.selected else 65)
                pen=QPen(color,1);pen.setCosmetic(True);pen.setStyle(Qt.PenStyle.DashLine)
                self.items.append(self.view.scene().addPath(path,pen,brush))
                path=QPainterPath(QPointF(*guide['points'][0]))
                for p in curve_samples(guide)[1:]:path.lineTo(QPointF(*p))
                pen=QPen(QColor('white') if i==self.selected else color,2);pen.setCosmetic(True)
                self.items.append(self.view.scene().addPath(path,pen))
            pen=QPen(QColor('white') if i==self.selected else color,2);pen.setCosmetic(True)
            for p in guide['points']:
                handle=self.view.scene().addEllipse(-4,-4,8,8,pen,color)
                handle.setFlag(handle.GraphicsItemFlag.ItemIgnoresTransformations,True)
                handle.setPos(QPointF(*p));self.items.append(handle)
            label=self.view.scene().addSimpleText(f"윤곽 {i+1} · ±{guide['half_width_px']:g}px")
            label.setBrush(color);label.setFlag(label.GraphicsItemFlag.ItemIgnoresTransformations,True)
            label.setPos(QPointF(*guide['points'][0])+QPointF(5,-20));self.items.append(label)
        for item in self.items:item.setZValue(500)

    def _draft_guide(self, cursor=None):
        points=deepcopy(self.draft)
        if cursor is not None and points and np.linalg.norm(cursor-points[-1])>.5:
            points.append(cursor.tolist())
        return dict(side=self.side,points=points,half_width_px=self.radius) if points else None

    def finish_curve(self, point=None):
        if point is not None and (not self.draft or np.linalg.norm(point-self.draft[-1])>1):
            self.draft.append(point.tolist())
        try:
            guide=normalize_guides([self._draft_guide()])[0]
        except (ValueError,TypeError) as exc:
            self.message.emit(str(exc));return
        self.guides.append(guide)
        self.set_drawing(False)
        self.selected=len(self.guides)-1
        self.render();self.changed.emit();self.selection_changed.emit()

    def eventFilter(self, watched, event):
        if not isValid(self.view):return False
        kind=event.type()
        if kind==QEvent.Type.KeyPress and self.view.hasFocus():
            if event.key()==Qt.Key.Key_Escape:
                self.set_drawing(False);self.render();return True
            if event.key() in (Qt.Key.Key_Return,Qt.Key.Key_Enter) and self.drawing and self.draft:
                self.finish_curve();return True
            if event.key()==Qt.Key.Key_Backspace and self.drawing and self.draft:
                self.draft.pop();self.render(self._draft_guide());return True
            if event.key()==Qt.Key.Key_Delete and self.selected>=0:
                self.remove_selected();return True
        if watched is not self.view.viewport() or self.view.overview:
            return False
        if kind==QEvent.Type.Wheel and not event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            if self.guides or self.drawing:
                index,_=self._hit(event,band=True) if not self.drawing else (-1,None)
                if index>=0 and index!=self.selected:self.select(index)
                steps=event.angleDelta().y()/120
                if steps:
                    current=self.guides[self.selected]['half_width_px'] if self.selected>=0 else self.radius
                    self.set_radius(float(np.clip(current+steps,1,300)))
                    self.selection_changed.emit()
                event.accept();return True
        if kind==QEvent.Type.MouseButtonDblClick and self.drawing and event.button()==Qt.MouseButton.LeftButton:
            self.finish_curve(self._point(event));return True
        if kind==QEvent.Type.MouseButtonPress and event.button()==Qt.MouseButton.LeftButton:
            if self.view.select_region or self.view.image_rect.isEmpty():return False
            point=self._point(event)
            if self.drawing:
                self.view.setFocus()
                if len(self.draft)>=200:
                    self.message.emit('한 곡선은 최대 200점입니다. 더블클릭으로 종료하세요.');return True
                if not self.draft or np.linalg.norm(point-self.draft[-1])>1:
                    self.draft.append(point.tolist())
                self.render(self._draft_guide());return True
            index,end=self._hit(event)
            if index>=0:
                self.view.setFocus();self.select(index)
                self.drag=('edit',point,(end,deepcopy(self.guides[index])))
                return True
        if kind==QEvent.Type.MouseMove and self.drawing and self.draft:
            self.render(self._draft_guide(self._point(event)));return True
        if kind==QEvent.Type.MouseMove and self.drag is not None:
            mode,start,stored=self.drag
            point=self._point(event)
            if mode=='edit':
                end,original=stored
                guide=deepcopy(original)
                if end is not None:
                    guide['points'][end]=point.tolist()
                else:
                    delta=point-start
                    for axis,limit in ((0,self.view.image_rect.width()-1),(1,self.view.image_rect.height()-1)):
                        coordinates=np.array(original['points'])[:,axis]
                        delta[axis]=np.clip(delta[axis],-coordinates.min(),limit-coordinates.max())
                    guide['points']=(np.array(original['points'])+delta).tolist()
                try:
                    self.guides[self.selected]=normalize_guides([guide])[0]
                except ValueError:
                    return True
                self.render()
            return True
        if kind==QEvent.Type.MouseButtonRelease and self.drag is not None:
            mode,start,stored=self.drag;self.drag=None
            self.render();self.changed.emit();self.selection_changed.emit();return True
        if kind==QEvent.Type.MouseButtonRelease and self.drawing:return True
        return False
