# MoodTune v3.8 — Báo cáo thay đổi so với v3.7

**Phiên bản:** `v3.8` (so với `v3.7` trong `BaoCao_MoodTune_v3.7.md`)
**Tên đầy đủ:** MoodTune — AI Cảm Xúc Tự Xây (RLUF Bandit · Knowledge Graph · Self-Attention) + Gợi ý nhạc Jamendo Hybrid
**Chủ đề nâng cấp:** Giao diện đa theme — thêm bộ chọn 3 theme (Tối · Midnight, Sáng · Aurora, Rực rỡ · Sunset) áp dụng ngay trong app, dọn anti-pattern gradient-text, chuẩn hoá contrast WCAG AA, đồng bộ màu sơ đồ tri thức AI theo theme đang chọn.

v3.8 là bản nâng cấp thuần giao diện (CSS/JS frontend) — **không đổi kiến trúc model, không cần retrain, tương thích ngược 100%** với `weights.npz` hiện có. Toàn bộ thay đổi nằm trong `frontend/index.html`.

---

## Thay đổi 1: Theme switcher — 3 theme chọn ngay trong app

### Mô tả
Trước v3.8, MoodTune chỉ có một giao diện tối cố định (Midnight), không có lựa chọn nào khác. Thêm bộ chọn giao diện với 3 theme:

- **Tối · Midnight** (mặc định) — nền tối, xanh Spotify-green & tím, dịu mắt khi nghe nhạc buổi tối.
- **Sáng · Aurora** — nền trắng thoáng, sương màu cam-tím-xanh-hồng loang nhẹ ở các góc như cực quang, card nổi bằng shadow.
- **Rực rỡ · Sunset** — gradient hoàng hôn chuyển động nhẹ phía sau, card kính mờ (glass/backdrop-blur), đổi hẳn tông cam-hồng-vàng.

Chọn theme nào thì áp dụng ngay (không cần reload), tự nhớ lựa chọn qua `localStorage`, và **không nhấp nháy sai theme** lúc tải trang (theme cũ được set trước cả khi CSS/JS chính load xong).

### Cài đặt kỹ thuật
- **Chống nhấp nháy (FOUC)**: một `<script>` inline ngay đầu `<head>` (trước khi parser chạm tới phần `<body>`) đọc `localStorage.getItem('mt_theme')` và set `data-theme` trên `<html>` ngay — chạy xong trước khi trình duyệt paint frame đầu, nên không có khoảnh khắc hiện sai theme rồi nhảy qua theme đúng.
- **Biến CSS theo theme**: mỗi theme định nghĩa lại toàn bộ token màu (`--bg`, `--surface`, `--card`, `--border`, `--accent`, `--accent2`, `--accent3`, `--green`, `--gold`, `--red`, `--text`, `--muted`, `--glow`) trong selector `html[data-theme="..."]` — mọi nơi trong CSS đã dùng `var(--token)` từ trước nên tự động đổi theo, không phải sửa lại từng rule.
- **JS điều khiển**: mảng `THEMES` (id/name/desc/swatches) render danh sách lựa chọn trong modal "🎨 Giao diện"; `applyTheme(id)` set `data-theme`, ghi `localStorage`, đồng bộ `<meta name="theme-color">` theo theme (để thanh trạng thái mobile đổi màu khớp); `renderThemeOptions()` vẽ lại danh sách với theme hiện tại được đánh dấu "đang dùng".
- Theme **Sunset** là một "full visual overhaul" thực sự, không phải chỉ recolor: card dùng `backdrop-filter: blur() saturate()` (kính mờ), border-radius lớn hơn, và hơn 20 rule riêng cho từng thành phần (nút, badge, toast...) để khớp tông cam-hồng.

### Kết quả kiểm thử
Dựng static server cho `frontend/`, dùng Playwright chụp cả 3 theme qua `localStorage.setItem('mt_theme', ...)` + reload — xác nhận cả 3 theme render đúng, không vỡ layout, chuyển theme tức thời không cần reload, và screenshot lúc tải trang (trước khi JS chính chạy) không lộ theme sai.

---

## Thay đổi 2: Dọn anti-pattern gradient-text, chuẩn hoá contrast WCAG AA

### Mô tả
Bản trước dùng `background-clip: text` để tô gradient (xanh→tím) cho logo/heading — pattern này đã lỗi thời, gây render mờ/răng cưa trên một số trình duyệt và **không đảm bảo contrast** so với nền nếu nền cũng đổi màu (như khi thêm theme Sáng, gradient-text trên nền trắng dễ rớt dưới ngưỡng AA ở phần màu nhạt của gradient).

### Cài đặt kỹ thuật
- Bỏ hẳn `background-clip: text` ở logo/heading, chuyển logo về `color: var(--accent2)` đặc — một màu rắn, luôn kiểm tra được tỉ lệ contrast so với `var(--bg)`/`var(--card)`.
- Re-tokenize lại bảng màu `--text`/`--muted`/`--accent*` cho cả 3 theme sao cho mọi cặp chữ/nền đạt tối thiểu **WCAG AA (4.5:1 với chữ thường, 3:1 với chữ lớn/đậm)** — đặc biệt theme Sáng phải đổi `--accent`/`--accent2`/`--green` sang sắc đậm hơn (`#127a3e`, `#0a6b34`) so với bản tối (`#1db954`, `#1ed760`) vì nền trắng cần độ đậm cao hơn mới đủ tương phản.

### Kết quả kiểm thử
Soát thủ công từng theme qua screenshot Playwright: chữ chính/phụ đều đọc rõ trên cả 3 nền (tối, trắng, gradient hoàng hôn có lớp card kính mờ nền tối phía sau để giữ contrast).

---

## Thay đổi 3: Sơ đồ tri thức AI (canvas) đọc màu theo theme

### Mô tả
Sơ đồ tri thức cảm xúc (giới thiệu từ v3.0) vẽ trực tiếp bằng Canvas 2D, trước đây **hardcode cố định** màu tia năng lượng và bong bóng cảm xúc (xanh `rgb(29,185,84)` / tím `rgb(124,106,247)`) — khi đổi sang theme Sáng hoặc Sunset, sơ đồ vẫn giữ đúng 2 màu xanh-tím cũ, lệch tông với phần còn lại của UI.

### Cài đặt kỹ thuật
- **`renderKnowledgeGraph()`**: trước khi vẽ, đọc `getComputedStyle(document.documentElement)` để lấy `--text` (màu chữ/đường nét), `--accent` (màu tia năng lượng + node trung tâm), `--accent3` (màu bong bóng cảm xúc vòng ngoài) — `hexToRgb()` chuyển hex từ CSS variable sang `rgb()` để dùng trong `ctx.fillStyle`/`ctx.strokeStyle` (Canvas 2D không hiểu trực tiếp `var(--token)`).
- Vẽ lại theo đúng theme đang active mỗi lần `renderKnowledgeGraph()` được gọi (sau mỗi lần phân tích cảm xúc mới) — không cần thêm logic nghe sự kiện đổi theme riêng vì hàm vẽ luôn đọc giá trị mới nhất tại thời điểm gọi.

### Kết quả kiểm thử
Đổi theme rồi phân tích lại một câu mẫu — tia/bong bóng trong sơ đồ đổi đúng theo bảng màu theme đang chọn (xanh-tím ở Midnight, xanh đậm-tím ở Aurora, cam-hồng ở Sunset).

---

## Thay đổi 4: Vá hiệu ứng nền Aurora (theme Sáng) không chạy, tăng độ đậm, chống lộ rìa

### Mô tả
Bản nháp đầu của theme Sáng thêm animation "trôi nhẹ" cho 4 quầng màu nền (giống cách Sunset đã có `sunsetDrift`), nhưng animate qua `background-position` trong khi mỗi `radial-gradient(... at center ...)` không khai báo `background-size` — ảnh gradient mặc định khít đúng bằng khung chứa, nên dịch `background-position` của một ảnh đã phủ kín 100% không tạo ra thay đổi gì: animation chạy nhưng **vô hình**, kiểm tra trực tiếp trên trình duyệt thấy nền đứng im.

Sau khi đổi sang `transform: translate()` để có chuyển động thật, lại phát sinh 2 vấn đề tinh chỉnh: (1) màu quá nhạt, khó nhận ra phần nào đang trôi; (2) khi tăng biên độ trôi lên, rìa cong của từng quầng màu (elip) lộ ra giữa trang — thấy rõ "đoạn cuối" hình oval của nó, mất tự nhiên.

### Cài đặt kỹ thuật
- **Đổi cơ chế animate**: bỏ `background-position`, animate `transform: translate()` trên cả `body::before` — luôn tạo chuyển động nhìn thấy được bất kể kích thước ảnh nền. Giữ lại toạ độ `at X% Y%` cố định trong từng `radial-gradient()` (đúng như layout tĩnh ban đầu), nới `inset` của lớp phủ ra `-10%` để có biên dự phòng quanh viewport khi dịch.
- **Tăng độ đậm**: alpha 4 quầng màu tăng từ `.14–.24` lên `.24–.40`, dễ nhận ra phần đang chuyển động hơn mà không phá contrast (lớp này nằm dưới `.card`/`.player-card` có nền đặc riêng).
- **Tăng biên độ trôi**: từ `translate(2.5%, 2%)` lên `translate(9%, 7%)` ở keyframe 50% — chuyển động rõ ràng hơn hẳn so với bản đầu gần như không nhận ra được.
- **Chống lộ rìa elip**: phình to kích thước từng `ellipse` (ví dụ `62% 50%` → `100% 80%`) để điểm "tàn dần hết màu" (`transparent`) của mỗi quầng luôn nằm ngoài khung nhìn ở mọi thời điểm animation, dù biên độ trôi đã tăng gấp đôi — chỉ còn thấy quầng sáng mềm ở 4 góc, không bao giờ thấy được hình oval đầy đủ của nó.
- Giữ nguyên `@media (prefers-reduced-motion: reduce)` tắt hẳn animation cho người dùng đã bật giảm chuyển động ở hệ điều hành.
- Đồng bộ lại mô tả changelog v3.8 (README.md, README.vi.md, modal Lịch sử phiên bản trong app) để khớp với hành vi mới của theme Sáng.

### Kết quả kiểm thử
Dựng static server + Playwright, theme Sáng, chụp ảnh lặp tại nhiều mốc trong chu kỳ animation 26s:
- So byte-for-byte ảnh chụp ở `t=0` và `t=13s` (giữa chu kỳ) — khác nhau, xác nhận có chuyển động thật (khác với bản `background-position` ban đầu, vốn cho ảnh giống y nhau 100%).
- Crop cố định góc trên-trái (tách khỏi phần nội dung động bên dưới do gọi API thất bại trên static server) — quầng cam thấy rõ dịch chuyển giữa 2 mốc thời gian.
- Full-page screenshot tại đúng điểm xa nhất của chu kỳ (translate đỉnh) — không còn thấy viền/hình oval lộ ra giữa trang, chỉ còn quầng sáng mềm tự nhiên ở 4 góc.
- `prefers-reduced-motion: reduce` qua `page.emulateMedia()` — xác nhận `animationName` trả về `none`, animation tắt đúng như thiết kế.
- Theme Tối và Sunset chụp lại song song — không bị ảnh hưởng bởi thay đổi riêng cho theme Sáng.

---

## Bảng so sánh tổng quan v3.7 vs v3.8

| Khía cạnh | v3.7 | v3.8 |
|---|---|---|
| Số theme giao diện | 1 (Tối cố định) | **3** — Tối · Midnight, Sáng · Aurora, Rực rỡ · Sunset |
| Chọn theme | Không có | **Modal "🎨 Giao diện"**, áp dụng ngay, nhớ qua `localStorage` |
| Nhấp nháy theme lúc tải trang | — | **Không** — set `data-theme` trước paint bằng inline script |
| Logo/heading | Gradient-text (`background-clip: text`) | **Màu rắn**, đảm bảo contrast theo theme |
| Contrast chữ/nền | Chỉ tối ưu cho nền tối | **WCAG AA** ở cả 3 theme |
| Sơ đồ tri thức AI (canvas) | Hardcode màu xanh-tím cố định | **Đọc theme đang chọn** (`--accent`, `--accent3`) |
| Nền theme Sáng | — | Quầng màu pastel **trôi nhẹ liên tục** (translate, không lộ rìa), tự tắt khi `prefers-reduced-motion` |
| Kiến trúc MLP / số class | Không đổi | Không đổi |
| Tương thích `weights.npz` cũ | — | **100%** (không đổi kiến trúc, không cần retrain) |
| Version trên UI | `v3.7` | **`v3.8`** + entry Changelog mới |

---

## Triển khai

- Commit liên quan: `3942a66` (thêm theme switcher 3 theme, dọn gradient-text, canvas đọc màu theme), `f5bce7b` (vá animation nền Aurora không chạy + tăng độ đậm màu), cộng các tinh chỉnh tiếp theo trong cùng phiên làm việc (tăng gấp đôi biên độ trôi, phình to ellipse chống lộ rìa, đồng bộ lại changelog).
- `GET /api/health` → `200 OK`, kiến trúc không đổi (`vocab_size: 4201`, `output_size: 10`, `feedback_count: 3551`) — xác nhận các thay đổi giao diện không ảnh hưởng tới state model đang học online.
- Toàn bộ thay đổi nằm trong `frontend/index.html` (CSS/JS thuần, không build step) — không cần restart backend, không cần migrate dữ liệu.

> **Định hướng tiếp theo:** có thể cân nhắc thêm theme thứ 4 hoặc cho người dùng tự chỉnh tốc độ/biên độ animation nền nếu có phản hồi muốn tắt hẳn hiệu ứng trôi (ngoài việc tự tắt theo `prefers-reduced-motion`).
