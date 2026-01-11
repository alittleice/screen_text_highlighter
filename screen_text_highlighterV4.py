# 实时OCR 并检索文本 标注，支持手动选择区域，支持主屏幕与副屏幕选择，支持高级搜索模式
# ┌────────────┐
# │ 屏幕截图   │
# └─────┬──────┘
#       ↓
# ┌────────────┐
# │ OCR 识别   │ pytesseract
# └─────┬──────┘
#       ↓
# ┌────────────┐
# │ 文本匹配   │ keyword in text
# └─────┬──────┘
#       ↓
# ┌────────────┐
# │ 坐标映射   │ OCR box
# └─────┬──────┘
#       ↓
# ┌────────────┐
# │ 高亮覆盖层 │ PyQt5
# └────────────┘


import sys
import time
import threading
import mss
import pytesseract
import cv2
import numpy as np
import json
import os
import re
from typing import Optional, Tuple, List, Dict, Set

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLineEdit, QPushButton,
    QLabel, QVBoxLayout, QHBoxLayout, QGridLayout,
    QCheckBox, QSpinBox, QComboBox, QMessageBox,
    QGroupBox, QTabWidget, QTextEdit, QRadioButton,
    QButtonGroup, QListWidget, QListWidgetItem,
    QSplitter, QFrame
)
from PyQt5.QtGui import QPainter, QColor, QPen, QFont, QBrush, QFontMetrics
from PyQt5.QtCore import Qt, QRect, QObject, pyqtSignal, QPoint, QTimer

# ========= 配置 Tesseract 路径 =========
pytesseract.pytesseract.tesseract_cmd = r"D:\Application\Tesseract-OCR\tesseract.exe"

# ========= 全局变量 =========
matched_boxes = []
search_keywords = []  # 改为列表，存储多个关键词
search_logic = "OR"  # 默认逻辑关系
lock = threading.Lock()
ocr_active = True
selected_regions = {}  # {monitor_index: (x, y, width, height)}
config_file = "ocr_config.json"
is_selecting_region = False
monitor_info = {}  # 存储显示器信息
matched_texts = {}  # 存储匹配的文本和位置信息


# ========= 信号桥（线程安全） =========
class SignalBridge(QObject):
    refresh_overlay = pyqtSignal()
    update_status = pyqtSignal(str)
    update_monitor_info = pyqtSignal(dict)
    update_search_results = pyqtSignal(list)  # 新增：更新搜索结果


signal_bridge = SignalBridge()


# ========= 高级搜索模型 =========
class AdvancedSearchModel:
    """高级搜索模型，支持多种搜索模式"""

    def __init__(self):
        self.keywords = []  # 关键词列表
        self.logic = "OR"  # 逻辑关系: OR, AND
        self.case_sensitive = False
        self.partial_match = True
        self.use_regex = False  # 是否使用正则表达式
        self.whole_word = False  # 是否匹配完整单词

    def add_keyword(self, keyword: str):
        """添加关键词"""
        keyword = keyword.strip()
        if keyword and keyword not in self.keywords:
            self.keywords.append(keyword)

    def remove_keyword(self, keyword: str):
        """移除关键词"""
        if keyword in self.keywords:
            self.keywords.remove(keyword)

    def clear_keywords(self):
        """清空关键词"""
        self.keywords.clear()

    def match_text(self, text: str) -> bool:
        """匹配文本，返回是否匹配"""
        if not self.keywords or not text:
            return False

        text_to_match = text if self.case_sensitive else text.lower()

        # 使用正则表达式
        if self.use_regex:
            try:
                for pattern in self.keywords:
                    flags = 0 if self.case_sensitive else re.IGNORECASE
                    if re.search(pattern, text, flags):
                        if self.logic == "OR":
                            return True
                    elif self.logic == "AND":
                        return False
                return self.logic == "AND"
            except re.error:
                # 正则表达式错误，回退到普通匹配
                pass

        # 普通匹配
        matches = []
        for keyword in self.keywords:
            keyword_to_match = keyword if self.case_sensitive else keyword.lower()

            if self.partial_match:
                if self.whole_word:
                    # 匹配完整单词
                    words = re.findall(r'\b\w+\b', text_to_match)
                    match = any(keyword_to_match == (word if self.case_sensitive else word.lower())
                                for word in words)
                else:
                    match = keyword_to_match in text_to_match
            else:
                match = keyword_to_match == text_to_match

            matches.append(match)

        # 根据逻辑关系判断
        if self.logic == "OR":
            return any(matches)
        else:  # AND
            return all(matches)

    def get_match_details(self, text: str) -> Dict:
        """获取匹配详情"""
        if not self.keywords or not text:
            return {"matched": False, "keywords": []}

        details = {
            "matched": False,
            "keywords": [],
            "text": text
        }

        for keyword in self.keywords:
            keyword_to_match = keyword if self.case_sensitive else keyword.lower()
            text_to_match = text if self.case_sensitive else text.lower()

            if self.partial_match:
                if self.whole_word:
                    words = re.findall(r'\b\w+\b', text_to_match)
                    match = any(keyword_to_match == (word if self.case_sensitive else word.lower())
                                for word in words)
                else:
                    match = keyword_to_match in text_to_match
            else:
                match = keyword_to_match == text_to_match

            if match:
                details["keywords"].append(keyword)
                details["matched"] = True if self.logic == "OR" else (details.get("matched", True))

        # 对于AND关系，需要所有关键词都匹配
        if self.logic == "AND" and len(details["keywords"]) < len(self.keywords):
            details["matched"] = False

        return details


class RegionSelectionWindow(QWidget):
    """区域选择窗口 - 支持多显示器"""
    region_selected = pyqtSignal(tuple, int)  # (x, y, width, height), monitor_index

    def __init__(self, target_monitor: int = None):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowStaysOnTopHint |
            Qt.FramelessWindowHint |
            Qt.Tool
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_NoSystemBackground)

        self.target_monitor = target_monitor  # 目标显示器索引
        self.start = QPoint()
        self.end = QPoint()
        self.is_selecting = False
        self.setCursor(Qt.CrossCursor)

        # 初始化显示器信息
        global monitor_info
        if not monitor_info:
            monitor_info = get_monitor_info()

        if target_monitor and target_monitor in monitor_info:
            # 只在目标显示器上显示选择窗口
            monitor = monitor_info[target_monitor]
            self.setGeometry(
                monitor['left'],
                monitor['top'],
                monitor['width'],
                monitor['height']
            )
        else:
            # 全屏显示（覆盖所有显示器）
            self.showFullScreen()

        # 状态提示
        self.status_label = QLabel(self)
        self.status_label.setStyleSheet("""
            QLabel {
                background-color: rgba(0, 0, 0, 180);
                color: white;
                padding: 10px;
                border-radius: 5px;
                font-size: 14px;
            }
        """)

        if target_monitor:
            self.status_label.setText(f"在显示器 {target_monitor} 上选择区域，按ESC键取消")
        else:
            self.status_label.setText("选择区域，按ESC键取消")

        self.status_label.adjustSize()
        self.status_label.move(20, 20)

        # 绘制显示器边界
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # 绘制半透明背景
        if self.target_monitor:
            # 只绘制目标显示器区域
            monitor = monitor_info[self.target_monitor]
            painter.fillRect(
                0, 0, monitor['width'], monitor['height'],
                QColor(0, 0, 0, 30)
            )
        else:
            # 绘制所有显示器区域
            for idx, monitor in monitor_info.items():
                x = monitor['left'] - (self.x() if hasattr(self, 'x') else 0)
                y = monitor['top'] - (self.y() if hasattr(self, 'y') else 0)
                painter.fillRect(
                    x, y, monitor['width'], monitor['height'],
                    QColor(0, 0, 0, 30)
                )

                # 绘制显示器边界和标签
                painter.setPen(QPen(QColor(255, 255, 255, 100), 2))
                painter.setBrush(Qt.NoBrush)
                painter.drawRect(x, y, monitor['width'], monitor['height'])

                painter.setPen(QColor(255, 255, 255))
                painter.setFont(QFont("Arial", 12, QFont.Bold))
                painter.drawText(x + 10, y + 30, f"显示器 {idx}")

        # 如果还没开始选择，显示提示文本
        if not self.is_selecting:
            painter.setPen(QColor(255, 255, 255))
            painter.setFont(QFont("Arial", 16, QFont.Bold))

            if self.target_monitor:
                painter.drawText(50, 100, f"请在显示器 {self.target_monitor} 上选择OCR区域")
            else:
                painter.drawText(50, 100, "请拖动鼠标选择OCR区域")

            painter.drawText(50, 140, "按ESC键取消选择")

        # 绘制选择区域
        if self.is_selecting and not self.start.isNull() and not self.end.isNull():
            x = min(self.start.x(), self.end.x())
            y = min(self.start.y(), self.end.y())
            w = abs(self.start.x() - self.end.x())
            h = abs(self.start.y() - self.end.y())

            # 绘制选择框
            painter.setPen(QPen(QColor(255, 0, 0), 2, Qt.SolidLine))
            painter.setBrush(QColor(255, 0, 0, 30))
            painter.drawRect(x, y, w, h)

            # 显示尺寸和所在显示器
            painter.setPen(QColor(255, 255, 255))
            painter.setFont(QFont("Arial", 10))

            # 确定选择区域所在的显示器
            center_x = x + w // 2
            center_y = y + h // 2
            monitor_idx = self.get_monitor_at_point(center_x, center_y)

            if monitor_idx:
                painter.drawText(x + w + 5, y - 5, f"{w} x {h} (显示器 {monitor_idx})")
            else:
                painter.drawText(x + w + 5, y - 5, f"{w} x {h}")

    def get_monitor_at_point(self, x: int, y: int) -> Optional[int]:
        """获取指定点所在的显示器索引"""
        global_x = x + (self.x() if hasattr(self, 'x') else 0)
        global_y = y + (self.y() if hasattr(self, 'y') else 0)

        for idx, monitor in monitor_info.items():
            if (monitor['left'] <= global_x <= monitor['right'] and
                    monitor['top'] <= global_y <= monitor['bottom']):
                return idx
        return None

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.start = event.pos()
            self.end = event.pos()
            self.is_selecting = True
            # 隐藏提示
            self.status_label.hide()
            self.update()

    def mouseMoveEvent(self, event):
        if self.is_selecting:
            self.end = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.is_selecting:
            x = min(self.start.x(), self.end.x())
            y = min(self.start.y(), self.end.y())
            w = abs(self.start.x() - self.end.x())
            h = abs(self.start.y() - self.end.y())

            # 最小区域检查
            if w < 50 or h < 30:
                QMessageBox.warning(self, "区域太小", "请选择一个更大的区域")
                return

            # 计算全局坐标
            global_x = x + (self.x() if hasattr(self, 'x') else 0)
            global_y = y + (self.y() if hasattr(self, 'y') else 0)

            # 确定区域所在的显示器
            monitor_idx = self.get_monitor_at_point(x + w // 2, y + h // 2)

            if not monitor_idx:
                QMessageBox.warning(self, "错误", "无法确定区域所在的显示器")
                return

            # 转换为相对于显示器的坐标
            monitor = monitor_info[monitor_idx]
            relative_x = global_x - monitor['left']
            relative_y = global_y - monitor['top']

            print(f"区域选择完成: 显示器 {monitor_idx}, 坐标: ({relative_x}, {relative_y}, {w}, {h})")

            # 发送信号，包含显示器索引
            self.region_selected.emit((relative_x, relative_y, w, h), monitor_idx)
            self.close()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            print("区域选择已取消")
            self.close()


class OverlayWindow(QWidget):
    """高亮覆盖窗口 - 支持多显示器"""

    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowStaysOnTopHint |
            Qt.FramelessWindowHint |
            Qt.Tool
        )
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_TranslucentBackground)

        # 获取显示器信息并设置全屏
        global monitor_info
        monitor_info = get_monitor_info()

        # 创建一个覆盖所有显示器的窗口
        min_x = min(m['left'] for m in monitor_info.values())
        min_y = min(m['top'] for m in monitor_info.values())
        max_right = max(m['right'] for m in monitor_info.values())
        max_bottom = max(m['bottom'] for m in monitor_info.values())

        self.setGeometry(min_x, min_y, max_right - min_x, max_bottom - min_y)
        self.show()

        # 存储区域边框
        self.region_rects = {}  # {monitor_index: QRect}

        # 颜色配置
        self.highlight_colors = [
            QColor(255, 255, 0, 90),  # 黄色
            QColor(0, 255, 0, 90),  # 绿色
            QColor(0, 200, 255, 90),  # 青色
            QColor(255, 100, 0, 90),  # 橙色
            QColor(255, 0, 255, 90)  # 紫色
        ]

        self.border_colors = [
            QColor(255, 0, 0, 180),  # 红色
            QColor(0, 150, 0, 180),  # 深绿
            QColor(0, 100, 200, 180),  # 深蓝
            QColor(200, 80, 0, 180),  # 棕色
            QColor(180, 0, 180, 180)  # 深紫
        ]

        signal_bridge.refresh_overlay.connect(self.update)

    def set_region(self, region: tuple, monitor_index: int):
        """设置指定显示器的OCR区域"""
        if region:
            # 转换为全局坐标
            monitor = monitor_info.get(monitor_index)
            if monitor:
                global_x = region[0] + monitor['left']
                global_y = region[1] + monitor['top']
                self.region_rects[monitor_index] = QRect(
                    global_x, global_y, region[2], region[3]
                )
        else:
            if monitor_index in self.region_rects:
                del self.region_rects[monitor_index]

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # 绘制所有显示器的区域边框
        for monitor_idx, rect in self.region_rects.items():
            # 只绘制在更新区域内的部分
            if event.rect().intersects(rect):
                # 不同显示器使用不同颜色
                colors = [
                    QColor(0, 255, 0, 180),  # 绿色 - 显示器1
                    QColor(0, 200, 255, 180),  # 青色 - 显示器2
                    QColor(255, 100, 0, 180),  # 橙色 - 显示器3
                    QColor(255, 0, 255, 180)  # 紫色 - 显示器4
                ]

                color_idx = (monitor_idx - 1) % len(colors)
                painter.setPen(QPen(colors[color_idx], 2, Qt.DashLine))
                painter.setBrush(Qt.NoBrush)
                painter.drawRect(rect)

                # 在区域左上角显示提示
                painter.setPen(colors[color_idx])
                painter.setFont(QFont("Arial", 10))
                painter.drawText(
                    rect.x(),
                    rect.y() - 10,
                    f"显示器 {monitor_idx}: {rect.width()} x {rect.height()}"
                )

        lock.acquire()
        for i, (x, y, w, h, monitor_idx, keyword_idx) in enumerate(matched_boxes):
            # 转换为全局坐标
            monitor = monitor_info.get(monitor_idx)
            if not monitor:
                continue

            global_x = x + monitor['left']
            global_y = y + monitor['top']

            # OCR 框扩展，增强"涂抹感"
            pad_x = 4
            pad_y = 2

            rect = QRect(
                global_x - pad_x,
                global_y - pad_y,
                w + pad_x * 2,
                h + pad_y * 2
            )

            # 只绘制在更新区域内的部分
            if event.rect().intersects(rect):
                # 根据关键词索引选择颜色
                color_idx = keyword_idx % len(self.highlight_colors)
                highlight_color = self.highlight_colors[color_idx]
                border_color = self.border_colors[color_idx]

                # ---- 1. 画荧光笔（无边框）----
                painter.setPen(Qt.NoPen)
                painter.setBrush(highlight_color)
                painter.drawRoundedRect(rect, 5, 5)

                # ---- 2. 画精确边框 ----
                painter.setBrush(Qt.NoBrush)
                painter.setPen(QPen(border_color, 2))
                painter.drawRoundedRect(rect, 5, 5)

                # 显示匹配的关键词索引（可选）
                if len(matched_boxes) < 10:  # 只在不拥挤时显示
                    painter.setPen(QColor(255, 255, 255))
                    painter.setFont(QFont("Arial", 8, QFont.Bold))
                    painter.drawText(rect.x() + 5, rect.y() + 15, f"#{keyword_idx + 1}")
        lock.release()


class ControlPanel(QWidget):
    """控制面板窗口 - 支持多显示器和高级搜索"""

    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowStaysOnTopHint)
        self.setWindowTitle("OCR高级搜索高亮 - 控制面板")
        self.resize(600, 700)  # 增大窗口以适应高级搜索
        self.move(50, 50)

        # 初始化搜索模型
        self.search_model = AdvancedSearchModel()

        self.init_ui()
        self.load_config()

        # 连接显示器信息更新信号
        signal_bridge.update_monitor_info.connect(self.update_monitor_info)
        signal_bridge.update_search_results.connect(self.update_search_results)

    def init_ui(self):
        main_layout = QVBoxLayout()

        # 创建标签页
        self.tab_widget = QTabWidget()

        # 第一页：高级搜索
        search_tab = QWidget()
        search_layout = QVBoxLayout()
        self.setup_search_tab(search_layout)
        search_tab.setLayout(search_layout)
        self.tab_widget.addTab(search_tab, "高级搜索")

        # 第二页：多显示器设置
        monitor_tab = QWidget()
        monitor_layout = QVBoxLayout()
        self.setup_monitor_tab(monitor_layout)
        monitor_tab.setLayout(monitor_layout)
        self.tab_widget.addTab(monitor_tab, "多显示器")

        # 第三页：OCR设置
        ocr_tab = QWidget()
        ocr_layout = QVBoxLayout()
        self.setup_ocr_tab(ocr_layout)
        ocr_tab.setLayout(ocr_layout)
        self.tab_widget.addTab(ocr_tab, "OCR设置")

        # 第四页：显示设置
        display_tab = QWidget()
        display_layout = QVBoxLayout()
        self.setup_display_tab(display_layout)
        display_tab.setLayout(display_layout)
        self.tab_widget.addTab(display_tab, "显示设置")

        main_layout.addWidget(self.tab_widget)

        # 搜索结果面板
        results_group = QGroupBox("搜索结果")
        results_layout = QVBoxLayout()

        # 搜索结果列表
        self.results_list = QListWidget()
        self.results_list.setMaximumHeight(150)
        results_layout.addWidget(self.results_list)

        # 结果统计
        self.results_stats = QLabel("找到 0 个匹配")
        self.results_stats.setStyleSheet("font-weight: bold; color: green;")
        results_layout.addWidget(self.results_stats)

        results_group.setLayout(results_layout)
        main_layout.addWidget(results_group)

        # 状态显示
        self.status_label = QLabel("状态: 准备就绪")
        self.status_label.setStyleSheet(
            "font-weight: bold; padding: 5px; border: 1px solid gray; background-color: #f0f0f0;")
        main_layout.addWidget(self.status_label)

        # 按钮组
        button_layout = QHBoxLayout()
        self.save_btn = QPushButton("保存设置")
        self.save_btn.clicked.connect(self.save_config)
        button_layout.addWidget(self.save_btn)

        self.refresh_btn = QPushButton("刷新显示器")
        self.refresh_btn.clicked.connect(self.refresh_monitors)
        button_layout.addWidget(self.refresh_btn)

        self.test_btn = QPushButton("测试搜索")
        self.test_btn.clicked.connect(self.test_search)
        button_layout.addWidget(self.test_btn)

        self.exit_btn = QPushButton("退出")
        self.exit_btn.clicked.connect(self.close_application)
        button_layout.addWidget(self.exit_btn)

        main_layout.addLayout(button_layout)

        # 信息标签
        self.info_label = QLabel("提示: 按Ctrl+S保存，Ctrl+R刷新显示器，Ctrl+Q退出")
        self.info_label.setStyleSheet("color: gray; font-size: 10px;")
        main_layout.addWidget(self.info_label)

        self.setLayout(main_layout)

        # 连接信号
        signal_bridge.update_status.connect(self.update_status)

        # 初始刷新显示器信息
        self.refresh_monitors()

        # 自动保存定时器
        self.auto_save_timer = QTimer()
        self.auto_save_timer.timeout.connect(self.auto_save_config)
        self.auto_save_timer.start(30000)  # 每30秒自动保存

    def setup_search_tab(self, layout):
        """设置高级搜索标签页"""
        # 关键词输入组
        keywords_group = QGroupBox("关键词设置")
        keywords_layout = QVBoxLayout()

        # 关键词输入框
        self.keywords_input = QTextEdit()
        self.keywords_input.setPlaceholderText("请输入要搜索的关键词，每行一个\n示例:\napple\nbanana\norange")
        self.keywords_input.setMaximumHeight(100)
        keywords_layout.addWidget(QLabel("关键词（每行一个）:"))
        keywords_layout.addWidget(self.keywords_input)

        # 快速添加按钮
        quick_buttons_layout = QHBoxLayout()
        self.add_btn = QPushButton("添加")
        self.add_btn.clicked.connect(self.add_keyword)
        quick_buttons_layout.addWidget(self.add_btn)

        self.clear_btn = QPushButton("清空")
        self.clear_btn.clicked.connect(self.clear_keywords)
        quick_buttons_layout.addWidget(self.clear_btn)

        self.load_example_btn = QPushButton("加载示例")
        self.load_example_btn.clicked.connect(self.load_example_keywords)
        quick_buttons_layout.addWidget(self.load_example_btn)

        keywords_layout.addLayout(quick_buttons_layout)
        keywords_group.setLayout(keywords_layout)
        layout.addWidget(keywords_group)

        # 搜索选项组
        options_group = QGroupBox("搜索选项")
        options_layout = QGridLayout()

        # 逻辑关系
        options_layout.addWidget(QLabel("逻辑关系:"), 0, 0)
        logic_layout = QHBoxLayout()
        self.or_radio = QRadioButton("OR (任一匹配)")
        self.and_radio = QRadioButton("AND (全部匹配)")
        self.or_radio.setChecked(True)
        logic_layout.addWidget(self.or_radio)
        logic_layout.addWidget(self.and_radio)

        self.logic_group = QButtonGroup()
        self.logic_group.addButton(self.or_radio)
        self.logic_group.addButton(self.and_radio)
        self.logic_group.buttonClicked.connect(self.update_logic)

        options_layout.addLayout(logic_layout, 0, 1)

        # 匹配选项
        options_layout.addWidget(QLabel("匹配选项:"), 1, 0)
        self.case_sensitive_check = QCheckBox("区分大小写")
        options_layout.addWidget(self.case_sensitive_check, 1, 1)

        self.partial_match_check = QCheckBox("部分匹配")
        self.partial_match_check.setChecked(True)
        options_layout.addWidget(self.partial_match_check, 2, 1)

        self.whole_word_check = QCheckBox("完整单词")
        options_layout.addWidget(self.whole_word_check, 3, 1)

        self.regex_check = QCheckBox("使用正则表达式")
        options_layout.addWidget(self.regex_check, 4, 1)

        # 连接信号
        self.case_sensitive_check.stateChanged.connect(self.update_search_model)
        self.partial_match_check.stateChanged.connect(self.update_search_model)
        self.whole_word_check.stateChanged.connect(self.update_search_model)
        self.regex_check.stateChanged.connect(self.update_search_model)

        options_group.setLayout(options_layout)
        layout.addWidget(options_group)

        # 当前关键词列表
        current_group = QGroupBox("当前关键词列表")
        current_layout = QVBoxLayout()

        self.keywords_list = QListWidget()
        self.keywords_list.setMaximumHeight(100)
        current_layout.addWidget(self.keywords_list)

        current_group.setLayout(current_layout)
        layout.addWidget(current_group)

        # 搜索按钮
        search_btn_layout = QHBoxLayout()
        self.search_btn = QPushButton("开始搜索")
        self.search_btn.clicked.connect(self.start_search)
        self.search_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 8px;")
        search_btn_layout.addWidget(self.search_btn)

        self.stop_btn = QPushButton("停止搜索")
        self.stop_btn.clicked.connect(self.stop_search)
        self.stop_btn.setStyleSheet("background-color: #f44336; color: white; padding: 8px;")
        search_btn_layout.addWidget(self.stop_btn)

        layout.addLayout(search_btn_layout)

        layout.addStretch()

    def setup_monitor_tab(self, layout):
        """设置多显示器标签页"""
        # 显示器选择组
        monitor_group = QGroupBox("显示器选择")
        monitor_group_layout = QVBoxLayout()

        self.monitor_info_label = QLabel("正在检测显示器...")
        self.monitor_info_label.setWordWrap(True)
        monitor_group_layout.addWidget(self.monitor_info_label)

        # 显示器选择按钮
        self.monitor_buttons_layout = QVBoxLayout()
        monitor_group_layout.addLayout(self.monitor_buttons_layout)

        monitor_group.setLayout(monitor_group_layout)
        layout.addWidget(monitor_group)

        # 区域管理组
        region_group = QGroupBox("区域管理")
        region_layout = QVBoxLayout()

        self.region_list_label = QLabel("已选择的区域:")
        region_layout.addWidget(self.region_list_label)

        self.region_list_widget = QLabel("无")
        self.region_list_widget.setWordWrap(True)
        self.region_list_widget.setStyleSheet("border: 1px solid gray; padding: 5px;")
        region_layout.addWidget(self.region_list_widget)

        # 区域操作按钮
        region_btn_layout = QHBoxLayout()
        self.clear_all_regions_btn = QPushButton("清除所有区域")
        self.clear_all_regions_btn.clicked.connect(self.clear_all_regions)
        region_btn_layout.addWidget(self.clear_all_regions_btn)

        self.select_all_monitors_btn = QPushButton("选择所有显示器")
        self.select_all_monitors_btn.clicked.connect(self.select_all_monitors)
        region_btn_layout.addWidget(self.select_all_monitors_btn)

        region_layout.addLayout(region_btn_layout)

        region_group.setLayout(region_layout)
        layout.addWidget(region_group)

        layout.addStretch()

    def setup_ocr_tab(self, layout):
        """设置OCR标签页"""
        # OCR设置组
        ocr_group = QGroupBox("OCR设置")
        ocr_layout = QGridLayout()

        ocr_layout.addWidget(QLabel("OCR语言:"), 0, 0)
        self.language_combo = QComboBox()
        languages = ["eng", "chi_sim", "eng+chi_sim", "jpn", "kor", "rus", "fra", "deu", "spa"]
        self.language_combo.addItems(languages)
        self.language_combo.setCurrentText("eng+chi_sim")
        ocr_layout.addWidget(self.language_combo, 0, 1)

        ocr_layout.addWidget(QLabel("刷新间隔(秒):"), 1, 0)
        self.interval_spin = QSpinBox()
        self.interval_spin.setRange(1, 30)
        self.interval_spin.setValue(2)
        self.interval_spin.valueChanged.connect(self.update_interval)
        ocr_layout.addWidget(self.interval_spin, 1, 1)

        ocr_layout.addWidget(QLabel("PSM模式:"), 2, 0)
        self.psm_combo = QComboBox()
        psm_options = [
            "3: 自动页面分割(默认)",
            "6: 假设为统一块",
            "7: 单行文本",
            "8: 单个单词",
            "11: 稀疏文本",
            "12: 稀疏文本带OSD"
        ]
        self.psm_combo.addItems(psm_options)
        self.psm_combo.setCurrentIndex(0)
        ocr_layout.addWidget(self.psm_combo, 2, 1)

        ocr_layout.addWidget(QLabel("OCR引擎模式:"), 3, 0)
        self.oem_combo = QComboBox()
        oem_options = [
            "0: 传统引擎",
            "1: 神经网络LSTM",
            "2: 传统+神经网络",
            "3: 默认"
        ]
        self.oem_combo.addItems(oem_options)
        self.oem_combo.setCurrentIndex(3)
        ocr_layout.addWidget(self.oem_combo, 3, 1)

        self.ocr_active_check = QCheckBox("启用OCR")
        self.ocr_active_check.setChecked(True)
        self.ocr_active_check.stateChanged.connect(self.toggle_ocr)
        ocr_layout.addWidget(self.ocr_active_check, 4, 0, 1, 2)

        # 性能选项
        self.fast_mode_check = QCheckBox("快速模式（降低准确性）")
        ocr_layout.addWidget(self.fast_mode_check, 5, 0, 1, 2)

        ocr_group.setLayout(ocr_layout)
        layout.addWidget(ocr_group)

        # 图像预处理组
        preprocess_group = QGroupBox("图像预处理")
        preprocess_layout = QGridLayout()

        preprocess_layout.addWidget(QLabel("图像缩放(%):"), 0, 0)
        self.scale_spin = QSpinBox()
        self.scale_spin.setRange(50, 200)
        self.scale_spin.setValue(100)
        self.scale_spin.setSuffix("%")
        preprocess_layout.addWidget(self.scale_spin, 0, 1)

        preprocess_layout.addWidget(QLabel("二值化阈值:"), 1, 0)
        self.threshold_combo = QComboBox()
        thresholds = ["自动", "固定127", "OTSU", "自适应"]
        self.threshold_combo.addItems(thresholds)
        self.threshold_combo.setCurrentText("自动")
        preprocess_layout.addWidget(self.threshold_combo, 1, 1)

        preprocess_layout.addWidget(QLabel("去噪强度:"), 2, 0)
        self.denoise_spin = QSpinBox()
        self.denoise_spin.setRange(0, 10)
        self.denoise_spin.setValue(3)
        preprocess_layout.addWidget(self.denoise_spin, 2, 1)

        preprocess_group.setLayout(preprocess_layout)
        layout.addWidget(preprocess_group)

        layout.addStretch()

    def setup_display_tab(self, layout):
        """设置显示标签页"""
        # 高亮设置组
        highlight_group = QGroupBox("高亮显示设置")
        highlight_layout = QGridLayout()

        highlight_layout.addWidget(QLabel("高亮颜色:"), 0, 0)
        self.highlight_color_combo = QComboBox()
        colors = ["黄色", "绿色", "蓝色", "粉色", "橙色", "红色", "青色", "紫色"]
        self.highlight_color_combo.addItems(colors)
        self.highlight_color_combo.setCurrentText("黄色")
        highlight_layout.addWidget(self.highlight_color_combo, 0, 1)

        highlight_layout.addWidget(QLabel("透明度(%):"), 1, 0)
        self.opacity_spin = QSpinBox()
        self.opacity_spin.setRange(10, 100)
        self.opacity_spin.setValue(90)
        self.opacity_spin.setSuffix("%")
        highlight_layout.addWidget(self.opacity_spin, 1, 1)

        highlight_layout.addWidget(QLabel("边框宽度:"), 2, 0)
        self.border_width_spin = QSpinBox()
        self.border_width_spin.setRange(1, 10)
        self.border_width_spin.setValue(2)
        highlight_layout.addWidget(self.border_width_spin, 2, 1)

        highlight_layout.addWidget(QLabel("边框样式:"), 3, 0)
        self.border_style_combo = QComboBox()
        border_styles = ["实线", "虚线", "点线", "点划线"]
        self.border_style_combo.addItems(border_styles)
        self.border_style_combo.setCurrentText("实线")
        highlight_layout.addWidget(self.border_style_combo, 3, 1)

        highlight_layout.addWidget(QLabel("圆角半径:"), 4, 0)
        self.corner_radius_spin = QSpinBox()
        self.corner_radius_spin.setRange(0, 20)
        self.corner_radius_spin.setValue(5)
        highlight_layout.addWidget(self.corner_radius_spin, 4, 1)

        highlight_group.setLayout(highlight_layout)
        layout.addWidget(highlight_group)

        # 显示选项组
        display_options_group = QGroupBox("显示选项")
        display_options_layout = QVBoxLayout()

        self.show_region_border_check = QCheckBox("显示OCR区域边框")
        self.show_region_border_check.setChecked(True)
        display_options_layout.addWidget(self.show_region_border_check)

        self.show_keyword_index_check = QCheckBox("显示关键词索引")
        self.show_keyword_index_check.setChecked(True)
        display_options_layout.addWidget(self.show_keyword_index_check)

        self.show_match_count_check = QCheckBox("显示匹配计数")
        self.show_match_count_check.setChecked(True)
        display_options_layout.addWidget(self.show_match_count_check)

        self.fade_effect_check = QCheckBox("淡入淡出效果")
        display_options_layout.addWidget(self.fade_effect_check)

        display_options_group.setLayout(display_options_layout)
        layout.addWidget(display_options_group)

        # 快捷键组
        shortcut_group = QGroupBox("快捷键")
        shortcut_layout = QVBoxLayout()

        shortcuts = [
            "Ctrl+S: 保存设置",
            "Ctrl+R: 刷新显示器",
            "Ctrl+Q: 退出程序",
            "Ctrl+F: 开始搜索",
            "Ctrl+E: 停止搜索",
            "Ctrl+1-9: 切换到对应显示器",
            "ESC: 取消区域选择"
        ]

        for shortcut in shortcuts:
            label = QLabel(f"• {shortcut}")
            shortcut_layout.addWidget(label)

        shortcut_group.setLayout(shortcut_layout)
        layout.addWidget(shortcut_group)

        layout.addStretch()

    def add_keyword(self):
        """添加关键词"""
        text = self.keywords_input.toPlainText().strip()
        if text:
            lines = text.split('\n')
            for line in lines:
                keyword = line.strip()
                if keyword:
                    self.search_model.add_keyword(keyword)
            self.update_keywords_list()
            self.keywords_input.clear()
            self.status_label.setText(f"状态: 已添加 {len(lines)} 个关键词")

    def clear_keywords(self):
        """清空关键词"""
        self.search_model.clear_keywords()
        self.update_keywords_list()
        self.status_label.setText("状态: 已清空所有关键词")

    def load_example_keywords(self):
        """加载示例关键词"""
        example = "apple\nbanana\norange\ntest\ndemo"
        self.keywords_input.setPlainText(example)
        self.status_label.setText("状态: 已加载示例关键词")

    def update_keywords_list(self):
        """更新关键词列表显示"""
        self.keywords_list.clear()
        for i, keyword in enumerate(self.search_model.keywords, 1):
            item = QListWidgetItem(f"{i}. {keyword}")
            self.keywords_list.addItem(item)

    def update_logic(self):
        """更新逻辑关系"""
        if self.or_radio.isChecked():
            self.search_model.logic = "OR"
        else:
            self.search_model.logic = "AND"
        self.status_label.setText(f"状态: 逻辑关系已设置为 {self.search_model.logic}")

    def update_search_model(self):
        """更新搜索模型设置"""
        self.search_model.case_sensitive = self.case_sensitive_check.isChecked()
        self.search_model.partial_match = self.partial_match_check.isChecked()
        self.search_model.whole_word = self.whole_word_check.isChecked()
        self.search_model.use_regex = self.regex_check.isChecked()

    def start_search(self):
        """开始搜索"""
        # 从输入框获取关键词
        text = self.keywords_input.toPlainText().strip()
        if text:
            self.search_model.clear_keywords()
            lines = text.split('\n')
            for line in lines:
                keyword = line.strip()
                if keyword:
                    self.search_model.add_keyword(keyword)
            self.update_keywords_list()

        if not self.search_model.keywords:
            QMessageBox.warning(self, "警告", "请输入至少一个关键词")
            return

        # 更新搜索模型设置
        self.update_search_model()
        self.update_logic()

        global ocr_active
        ocr_active = True
        self.ocr_active_check.setChecked(True)

        self.status_label.setText(f"状态: 开始搜索 {len(self.search_model.keywords)} 个关键词 ({self.search_model.logic})")
        print(f"开始搜索: {self.search_model.keywords}")

    def stop_search(self):
        """停止搜索"""
        global ocr_active
        ocr_active = False
        self.ocr_active_check.setChecked(False)
        self.status_label.setText("状态: 搜索已停止")

    def test_search(self):
        """测试搜索"""
        test_text = "This is a test with apple and banana and orange"
        result = self.search_model.match_text(test_text)
        details = self.search_model.get_match_details(test_text)

        if result:
            msg = f"测试通过！\n文本: {test_text}\n匹配的关键词: {', '.join(details['keywords'])}"
            QMessageBox.information(self, "测试结果", msg)
        else:
            QMessageBox.warning(self, "测试结果", f"测试失败！\n文本: {test_text}\n未匹配任何关键词")

    def update_search_results(self, results):
        """更新搜索结果"""
        self.results_list.clear()
        count = 0

        for result in results:
            monitor_idx = result.get('monitor', 1)
            text = result.get('text', '').strip()
            keywords = result.get('keywords', [])

            if text and keywords:
                item_text = f"显示器{monitor_idx}: {text[:50]}{'...' if len(text) > 50 else ''}"
                if keywords:
                    item_text += f" [匹配: {', '.join(keywords[:3])}{'...' if len(keywords) > 3 else ''}]"

                item = QListWidgetItem(item_text)
                item.setToolTip(f"完整文本: {text}\n匹配关键词: {', '.join(keywords)}")
                self.results_list.addItem(item)
                count += 1

        self.results_stats.setText(f"找到 {count} 个匹配")

        if count > 0:
            self.results_stats.setStyleSheet("font-weight: bold; color: green;")
        else:
            self.results_stats.setStyleSheet("font-weight: bold; color: red;")

    def refresh_monitors(self):
        """刷新显示器信息"""
        global monitor_info, selected_regions
        monitor_info = get_monitor_info()

        # 更新显示器信息显示
        info_text = f"检测到 {len(monitor_info)} 个显示器:\n"
        for idx, monitor in monitor_info.items():
            info_text += f"显示器 {idx}: {monitor['width']}x{monitor['height']} "
            info_text += f"位置({monitor['left']}, {monitor['top']})\n"

        self.monitor_info_label.setText(info_text)

        # 清空并重新创建显示器按钮
        for i in reversed(range(self.monitor_buttons_layout.count())):
            widget = self.monitor_buttons_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()

        # 为每个显示器创建按钮
        for idx in monitor_info.keys():
            btn_text = f"在显示器 {idx} 上选择区域"
            if idx in selected_regions:
                region = selected_regions[idx]
                btn_text += f" (已选: {region[2]}x{region[3]})"

            btn = QPushButton(btn_text)
            btn.clicked.connect(lambda checked, m=idx: self.start_region_selection(m))
            self.monitor_buttons_layout.addWidget(btn)

        # 更新区域列表
        self.update_region_list()

        # 发送信号通知其他组件
        signal_bridge.update_monitor_info.emit(monitor_info)

        self.status_label.setText(f"状态: 已刷新 {len(monitor_info)} 个显示器")

    def update_monitor_info(self, info):
        """更新显示器信息"""
        global monitor_info
        monitor_info = info

    def start_region_selection(self, monitor_index: int):
        """启动指定显示器的区域选择"""
        global is_selecting_region
        if is_selecting_region:
            return

        is_selecting_region = True
        self.status_label.setText(f"状态: 请在显示器 {monitor_index} 上选择区域...")

        self.region_window = RegionSelectionWindow(monitor_index)
        self.region_window.region_selected.connect(self.on_region_selected)
        self.region_window.destroyed.connect(self.on_region_window_closed)
        self.region_window.show()

    def on_region_selected(self, region: tuple, monitor_index: int):
        """区域选择完成"""
        global selected_regions
        selected_regions[monitor_index] = region

        # 更新区域列表显示
        self.update_region_list()

        # 更新覆盖窗口的区域显示
        if overlay_window:
            overlay_window.set_region(region, monitor_index)
            overlay_window.update()

        self.status_label.setText(f"状态: 显示器 {monitor_index} 区域已选择")
        print(f"显示器 {monitor_index} OCR区域已设置为: {region}")

    def on_region_window_closed(self):
        """区域选择窗口关闭"""
        global is_selecting_region
        is_selecting_region = False
        self.status_label.setText("状态: 准备就绪")

    def update_region_list(self):
        """更新区域列表显示"""
        if not selected_regions:
            self.region_list_widget.setText("无")
            return

        text = ""
        for monitor_idx, region in selected_regions.items():
            if region:
                text += f"显示器 {monitor_idx}: ({region[0]}, {region[1]}, {region[2]}, {region[3]})\n"

        self.region_list_widget.setText(text)

    def clear_all_regions(self):
        """清除所有区域设置"""
        global selected_regions
        selected_regions.clear()

        # 更新覆盖窗口
        if overlay_window:
            for idx in monitor_info.keys():
                overlay_window.set_region(None, idx)
            overlay_window.update()

        self.update_region_list()
        self.status_label.setText("状态: 所有区域已清除")

    def select_all_monitors(self):
        """选择所有显示器（全屏）"""
        global selected_regions, monitor_info

        for idx in monitor_info.keys():
            monitor = monitor_info[idx]
            selected_regions[idx] = (0, 0, monitor['width'], monitor['height'])

            # 更新覆盖窗口
            if overlay_window:
                overlay_window.set_region(selected_regions[idx], idx)

        self.update_region_list()
        overlay_window.update()
        self.status_label.setText(f"状态: 已选择所有 {len(monitor_info)} 个显示器（全屏）")

    def toggle_ocr(self, state):
        """切换OCR激活状态"""
        global ocr_active
        ocr_active = (state == Qt.Checked)
        status = "启用" if ocr_active else "暂停"
        self.status_label.setText(f"状态: OCR已{status}")

    def update_interval(self):
        """更新刷新间隔"""
        self.status_label.setText(f"状态: 刷新间隔设置为 {self.interval_spin.value()} 秒")

    def update_status(self, message):
        """更新状态信息"""
        self.status_label.setText(f"状态: {message}")

    def auto_save_config(self):
        """自动保存配置"""
        try:
            self.save_config(silent=True)
        except:
            pass

    def save_config(self, silent=False):
        """保存配置到文件"""
        # 将区域转换为可序列化的格式
        serializable_regions = {}
        for monitor_idx, region in selected_regions.items():
            if region:
                serializable_regions[monitor_idx] = list(region)

        # 获取当前关键词
        keywords_text = self.keywords_input.toPlainText().strip()
        if not keywords_text and self.search_model.keywords:
            keywords_text = "\n".join(self.search_model.keywords)

        config = {
            'regions': serializable_regions,
            'keywords': keywords_text,
            'logic': "OR" if self.or_radio.isChecked() else "AND",
            'case_sensitive': self.case_sensitive_check.isChecked(),
            'partial_match': self.partial_match_check.isChecked(),
            'whole_word': self.whole_word_check.isChecked(),
            'use_regex': self.regex_check.isChecked(),
            'language': self.language_combo.currentText(),
            'interval': self.interval_spin.value(),
            'psm': self.psm_combo.currentIndex(),
            'oem': self.oem_combo.currentIndex(),
            'highlight_color': self.highlight_color_combo.currentText(),
            'opacity': self.opacity_spin.value(),
            'border_width': self.border_width_spin.value(),
            'border_style': self.border_style_combo.currentText(),
            'corner_radius': self.corner_radius_spin.value(),
            'scale': self.scale_spin.value(),
            'threshold': self.threshold_combo.currentText(),
            'denoise': self.denoise_spin.value(),
            'fast_mode': self.fast_mode_check.isChecked(),
            'show_region_border': self.show_region_border_check.isChecked(),
            'show_keyword_index': self.show_keyword_index_check.isChecked(),
            'show_match_count': self.show_match_count_check.isChecked(),
            'fade_effect': self.fade_effect_check.isChecked()
        }

        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            if not silent:
                self.status_label.setText("状态: 配置已保存")
                print(f"配置已保存到: {config_file}")
        except Exception as e:
            if not silent:
                QMessageBox.warning(self, "保存失败", f"无法保存配置: {str(e)}")

    def load_config(self):
        """从文件加载配置"""
        global selected_regions

        if os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)

                # 加载区域
                if 'regions' in config:
                    for monitor_idx_str, region in config['regions'].items():
                        monitor_idx = int(monitor_idx_str)
                        selected_regions[monitor_idx] = tuple(region)

                # 加载关键词
                keywords_text = config.get('keywords', '')
                self.keywords_input.setPlainText(keywords_text)

                # 更新搜索模型
                if keywords_text:
                    self.search_model.clear_keywords()
                    lines = keywords_text.split('\n')
                    for line in lines:
                        keyword = line.strip()
                        if keyword:
                            self.search_model.add_keyword(keyword)
                    self.update_keywords_list()

                # 加载搜索选项
                logic = config.get('logic', 'OR')
                if logic == "AND":
                    self.and_radio.setChecked(True)
                else:
                    self.or_radio.setChecked(True)

                self.case_sensitive_check.setChecked(config.get('case_sensitive', False))
                self.partial_match_check.setChecked(config.get('partial_match', True))
                self.whole_word_check.setChecked(config.get('whole_word', False))
                self.regex_check.setChecked(config.get('use_regex', False))

                # 更新搜索模型
                self.update_search_model()
                self.update_logic()

                # 加载OCR设置
                self.language_combo.setCurrentText(config.get('language', 'eng+chi_sim'))
                self.interval_spin.setValue(config.get('interval', 2))
                self.psm_combo.setCurrentIndex(config.get('psm', 0))
                self.oem_combo.setCurrentIndex(config.get('oem', 3))
                self.scale_spin.setValue(config.get('scale', 100))
                self.threshold_combo.setCurrentText(config.get('threshold', '自动'))
                self.denoise_spin.setValue(config.get('denoise', 3))
                self.fast_mode_check.setChecked(config.get('fast_mode', False))

                # 加载显示设置
                self.highlight_color_combo.setCurrentText(config.get('highlight_color', '黄色'))
                self.opacity_spin.setValue(config.get('opacity', 90))
                self.border_width_spin.setValue(config.get('border_width', 2))
                self.border_style_combo.setCurrentText(config.get('border_style', '实线'))
                self.corner_radius_spin.setValue(config.get('corner_radius', 5))
                self.show_region_border_check.setChecked(config.get('show_region_border', True))
                self.show_keyword_index_check.setChecked(config.get('show_keyword_index', True))
                self.show_match_count_check.setChecked(config.get('show_match_count', True))
                self.fade_effect_check.setChecked(config.get('fade_effect', False))

                if not silent:
                    self.status_label.setText("状态: 配置已加载")
                    print(f"配置已从 {config_file} 加载")

                # 更新区域列表
                self.update_region_list()

            except Exception as e:
                print(f"加载配置失败: {str(e)}")

    def close_application(self):
        """关闭应用程序"""
        reply = QMessageBox.question(
            self, '确认退出',
            '确定要退出程序吗？',
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            self.save_config()
            QApplication.quit()


def get_monitor_info() -> Dict[int, Dict]:
    """获取所有显示器信息"""
    info = {}
    with mss.mss() as sct:
        for i, monitor in enumerate(sct.monitors[1:], 1):  # 跳过第一个（虚拟桌面）
            info[i] = {
                'index': i,
                'left': monitor['left'],
                'top': monitor['top'],
                'width': monitor['width'],
                'height': monitor['height'],
                'right': monitor['left'] + monitor['width'],
                'bottom': monitor['top'] + monitor['height']
            }
    return info


def ocr_worker():
    """OCR工作线程 - 支持多显示器和高级搜索"""
    global matched_boxes, ocr_active, selected_regions, monitor_info, matched_texts

    with mss.mss() as sct:
        # 获取所有显示器
        monitors = sct.monitors[1:]  # 跳过第一个（虚拟桌面）

        while True:
            if not ocr_active:
                time.sleep(0.5)
                continue

            try:
                boxes = []
                search_results = []
                matched_texts.clear()

                # 对每个配置了区域的显示器进行处理
                for monitor_idx, region in selected_regions.items():
                    if not region:
                        continue

                    # 检查显示器索引是否有效
                    if monitor_idx - 1 >= len(monitors):
                        print(f"警告: 显示器 {monitor_idx} 不存在")
                        continue

                    monitor = monitors[monitor_idx - 1]

                    # 如果指定了区域，使用区域截图，否则使用整个显示器
                    if region:
                        # 区域相对于显示器的坐标
                        region_dict = {
                            'left': monitor['left'] + region[0],
                            'top': monitor['top'] + region[1],
                            'width': region[2],
                            'height': region[3]
                        }
                        screenshot_area = region_dict
                    else:
                        screenshot_area = monitor

                    try:
                        # 截图
                        img = np.array(sct.grab(screenshot_area))

                        # 图像预处理
                        # 1. 转换为灰度图
                        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

                        # 2. 去噪
                        denoise_strength = control_panel.denoise_spin.value()
                        if denoise_strength > 0:
                            gray = cv2.medianBlur(gray, denoise_strength * 2 + 1)

                        # 3. 缩放（如果设置了缩放）
                        scale_factor = control_panel.scale_spin.value() / 100.0
                        if scale_factor != 1.0:
                            new_width = int(gray.shape[1] * scale_factor)
                            new_height = int(gray.shape[0] * scale_factor)
                            gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_AREA)

                        # 4. 二值化
                        threshold_method = control_panel.threshold_combo.currentText()
                        if threshold_method == "固定127":
                            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
                        elif threshold_method == "OTSU":
                            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        elif threshold_method == "自适应":
                            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                           cv2.THRESH_BINARY, 11, 2)
                        else:  # 自动
                            binary = gray

                        # 获取OCR设置
                        language = control_panel.language_combo.currentText()
                        psm = control_panel.psm_combo.currentText().split(":")[0]
                        oem = control_panel.oem_combo.currentText().split(":")[0]

                        # 构建OCR配置
                        config_str = f'--psm {psm} --oem {oem}'
                        if control_panel.fast_mode_check.isChecked():
                            config_str += ' -c tessedit_do_invert=0'

                        # OCR识别
                        data = pytesseract.image_to_data(
                            binary,
                            output_type=pytesseract.Output.DICT,
                            lang=language,
                            config=config_str
                        )

                        # 处理匹配结果
                        search_model = control_panel.search_model

                        # 如果没有关键词，跳过
                        if not search_model.keywords:
                            continue

                        for i, text in enumerate(data["text"]):
                            text = text.strip()
                            if not text:
                                continue

                            # 使用高级搜索模型匹配
                            if search_model.match_text(text):
                                # 获取匹配详情
                                details = search_model.get_match_details(text)

                                if details['matched']:
                                    # 计算原始坐标（考虑缩放）
                                    x = int(data["left"][i] / scale_factor) + (region[0] if region else 0)
                                    y = int(data["top"][i] / scale_factor) + (region[1] if region else 0)
                                    w = int(data["width"][i] / scale_factor)
                                    h = int(data["height"][i] / scale_factor)

                                    # 找出匹配的关键词索引
                                    keyword_indices = []
                                    for keyword in details['keywords']:
                                        if keyword in search_model.keywords:
                                            keyword_indices.append(search_model.keywords.index(keyword))

                                    # 使用第一个匹配的关键词索引
                                    keyword_idx = keyword_indices[0] if keyword_indices else 0

                                    # 添加匹配框和显示器索引、关键词索引
                                    boxes.append((x, y, w, h, monitor_idx, keyword_idx))

                                    # 保存匹配文本信息
                                    match_id = f"{monitor_idx}_{x}_{y}"
                                    matched_texts[match_id] = {
                                        'text': text,
                                        'keywords': details['keywords'],
                                        'monitor': monitor_idx,
                                        'x': x,
                                        'y': y,
                                        'w': w,
                                        'h': h
                                    }

                                    # 添加到搜索结果
                                    search_results.append({
                                        'text': text,
                                        'keywords': details['keywords'],
                                        'monitor': monitor_idx
                                    })

                                    # 实时显示匹配信息
                                    if len(search_results) <= 5:  # 只显示前5个匹配
                                        keyword_text = ', '.join(details['keywords'][:3])
                                        keyword_text += '...' if len(details['keywords']) > 3 else ''
                                        signal_bridge.update_status.emit(
                                            f"显示器{monitor_idx}: 找到: {text[:20]}... [{keyword_text}]")

                    except Exception as e:
                        print(f"显示器 {monitor_idx} OCR处理错误: {str(e)}")

                # 更新匹配框
                lock.acquire()
                matched_boxes = boxes
                lock.release()

                # 发送搜索结果更新信号
                signal_bridge.update_search_results.emit(search_results)

                # 刷新覆盖层
                signal_bridge.refresh_overlay.emit()

                # 输出统计信息
                if boxes:
                    print(f"找到 {len(boxes)} 个匹配项")

            except Exception as e:
                print(f"主OCR处理错误: {str(e)}")
                signal_bridge.update_status.emit(f"错误: {str(e)}")

            # 等待下一次扫描
            interval = control_panel.interval_spin.value()
            time.sleep(interval)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    print("=== OCR高级搜索高亮程序 ===")
    print("支持多显示器、多关键词、逻辑关系搜索")
    print("程序启动中...")

    # 初始获取显示器信息
    monitor_info = get_monitor_info()
    print(f"检测到 {len(monitor_info)} 个显示器")

    # 创建覆盖窗口
    overlay_window = OverlayWindow()

    # 创建控制面板
    control_panel = ControlPanel()
    control_panel.show()

    # 启动OCR线程
    ocr_thread = threading.Thread(target=ocr_worker, daemon=True)
    ocr_thread.start()

    print("程序启动完成")
    print("使用说明:")
    print("1. 在'高级搜索'标签页输入多个关键词（每行一个）")
    print("2. 选择逻辑关系（OR/AND）")
    print("3. 在'多显示器'标签页选择OCR区域")
    print("4. 点击'开始搜索'按钮")
    print("=" * 50)

    sys.exit(app.exec_())