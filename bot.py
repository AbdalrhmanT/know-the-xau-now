# bot.py
import os
import sqlite3
from datetime import datetime, timedelta, timezone
import traceback

import telebot
from telebot import types

import pandas as pd
import numpy as np

# Optional: tvDatafeed (TradingView). تأكد من تثبيته وضبطه إن لزم
try:
    from tvDatafeed import TvDatafeed, Interval as TvdfInterval
    tv = TvDatafeed()
except Exception as e:
    tv = None
    TvdfInterval = None

# ========== إعدادات - عدل هنا حسب حاجتك ==========
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8299388128:AAFqd0uLD1ITDiBBZXD6bSC2DaRkksGt0JU")
ADMIN_ID = int(os.getenv("ADMIN_ID", "7500525431"))  # ضع أيدي الأدمن هنا
WALLET_ADDRESS = os.getenv("WALLET_ADDRESS", "UQDb-cfThq9yZgx0ls-PQWMRFkLD65u5cdy3ue6EjIVoivsE")
SUBSCRIPTION_PRICE = 15.0
CASHBACK_PER_PURCHASE = 2.0
DB_PATH = "bot.db"
SYMBOL = "XAUUSD"
EXCHANGE = "FX_IDC"  # جرب OANDA أو TVC لو احتجت
# =================================================

bot = telebot.TeleBot(BOT_TOKEN)


# -------------------- DB init --------------------
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY,
                capital REAL DEFAULT NULL,
                risk_percent REAL DEFAULT NULL,
                subscription_end TEXT DEFAULT NULL,
                creator_balance REAL DEFAULT 0
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS creators_codes (
                code TEXT PRIMARY KEY,
                discount_percent REAL,
                creator_id INTEGER
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS code_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                code TEXT,
                used_at TEXT,
                purchased INTEGER DEFAULT 0,
                amount REAL DEFAULT 0
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS withdraw_requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                creator_id INTEGER,
                amount REAL,
                wallet_address TEXT,
                status TEXT DEFAULT 'pending',
                requested_at TEXT
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS subscription_payments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                amount REAL,
                code TEXT DEFAULT NULL,
                status TEXT DEFAULT 'pending',
                requested_at TEXT
            )
        """)
        conn.commit()


init_db()


# -------------------- DB helpers --------------------
def ensure_user(user_id: int):
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("INSERT OR IGNORE INTO users (user_id) VALUES (?)", (user_id,))
        conn.commit()


def set_capital(user_id: int, capital: float):
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("UPDATE users SET capital=? WHERE user_id=?", (capital, user_id))
        conn.commit()


def set_risk(user_id: int, risk_percent: float):
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("UPDATE users SET risk_percent=? WHERE user_id=?", (risk_percent, user_id))
        conn.commit()


def get_user(user_id: int):
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("SELECT capital, risk_percent, subscription_end, creator_balance FROM users WHERE user_id=?", (user_id,))
        row = c.fetchone()
        if not row:
            return {"capital": None, "risk_percent": None, "subscription_end": None, "creator_balance": 0.0}
        capital, risk_percent, subscription_end, creator_balance = row
        return {
            "capital": capital,
            "risk_percent": risk_percent,
            "subscription_end": subscription_end,
            "creator_balance": creator_balance or 0.0
        }


def set_subscription_for_user(user_id: int, months: int = 1):
    end = (datetime.now(timezone.utc) + timedelta(days=30 * months)).date().isoformat()
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("UPDATE users SET subscription_end=? WHERE user_id=?", (end, user_id))
        conn.commit()
    return end


# creators codes
def create_creator_code(code: str, discount_percent: float, creator_id: int):
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("INSERT OR REPLACE INTO creators_codes (code, discount_percent, creator_id) VALUES (?, ?, ?)",
                  (code, discount_percent, creator_id))
        conn.commit()


def get_creator_code(code: str):
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("SELECT code, discount_percent, creator_id FROM creators_codes WHERE code=?", (code,))
        return c.fetchone()


def add_code_usage(user_id: int, code: str, purchased: int, amount: float):
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("INSERT INTO code_usage (user_id, code, used_at, purchased, amount) VALUES (?, ?, ?, ?, ?)",
                  (user_id, code, datetime.now(timezone.utc).isoformat(), purchased, amount))
        conn.commit()


def get_code_stats_by_creator(creator_id: int):
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("SELECT code FROM creators_codes WHERE creator_id=?", (creator_id,))
        row = c.fetchone()
        if not row:
            return None
        code = row[0]
        c.execute("SELECT COUNT(DISTINCT user_id), SUM(purchased), SUM(amount) FROM code_usage WHERE code=?", (code,))
        stats = c.fetchone()
        return code, stats  # stats = (unique_users, sum_purchased, sum_amount)


def credit_creator(creator_id: int, amount: float):
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("UPDATE users SET creator_balance = COALESCE(creator_balance,0) + ? WHERE user_id=?", (amount, creator_id))
        conn.commit()


def save_withdraw_request(creator_id: int, amount: float, wallet: str):
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("INSERT INTO withdraw_requests (creator_id, amount, wallet_address, requested_at) VALUES (?, ?, ?, ?)",
                  (creator_id, amount, wallet, datetime.now(timezone.utc).isoformat()))
        conn.commit()


def create_subscription_payment(user_id: int, amount: float, code: str = None):
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("INSERT INTO subscription_payments (user_id, amount, code, status, requested_at) VALUES (?, ?, ?, 'pending', ?)",
                  (user_id, amount, code, datetime.now(timezone.utc).isoformat()))
        pid = c.lastrowid
        conn.commit()
        return pid


def fetch_pending_sub_payments():
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("SELECT id, user_id, amount, code, requested_at FROM subscription_payments WHERE status='pending'")
        return c.fetchall()


def set_sub_payment_status(payment_id: int, status: str):
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("UPDATE subscription_payments SET status=? WHERE id=?", (status, payment_id))
        conn.commit()


# -------------------- Price fetch + SMC simplified --------------------
def fetch_prices_safe(symbol=SYMBOL, exchange=EXCHANGE, interval=None, bars=300, message=None):
    """جلب الأسعار من tvDatafeed مع معالجة الأخطاء"""
    if tv is None:
        if message:
            bot.reply_to(message, "⚠️ tvDatafeed غير مُهيأ على هذا النظام. تأكد من تثبيته/تسجيل الدخول.")
        raise RuntimeError("tvDatafeed not available")
    if interval is None:
        interval = TvdfInterval.in_30_minute
    try:
        df = tv.get_hist(symbol=symbol, exchange=exchange, interval=interval, n_bars=bars)
        if df is None or df.empty:
            raise RuntimeError("No data from tvDatafeed")
        # ensure numeric types
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        if 'volume' in df.columns:
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)
        else:
            df['volume'] = 0
        return df
    except Exception as e:
        if message:
            bot.reply_to(message, f"⚠️ خطأ في جلب الأسعار: {e}")
        raise


def smc_find_swings(closes, window=5):
    highs = []
    lows = []
    n = len(closes)
    for i in range(window, n - window):
        segment = closes[i - window:i + window + 1]
        center = closes[i]
        if center == max(segment):
            highs.append((i, float(center)))
        if center == min(segment):
            lows.append((i, float(center)))
    return highs, lows


def smc_determine_side_from_swings(highs, lows):
    if not highs or not lows:
        return None
    last_high = highs[-1][1]
    last_low = lows[-1][1]
    prev_high = highs[-2][1] if len(highs) >= 2 else None
    prev_low = lows[-2][1] if len(lows) >= 2 else None
    if prev_high and prev_low:
        if last_high > prev_high and last_low > prev_low:
            return "BUY"
        if last_high < prev_high and last_low < prev_low:
            return "SELL"
    return None


def smc_nearest_support_buy(lows):
    return lows[-1][1] if lows else None


def smc_nearest_resistance_sell(highs):
    return highs[-1][1] if highs else None


def analyze_tf(symbol, exchange, interval, bars=300, window=5, message=None):
    """
    يحلل فريم واحد: يرجع (side, sl_price, df, info) أو (None,None,None,None) on error
    info يحتوي متوسط حجم التداول recent_volume_avg للمساعدة في الفلترة
    """
    try:
        df = fetch_prices_safe(symbol=symbol, exchange=exchange, interval=interval, bars=bars, message=message)
    except Exception:
        return None, None, None, None

    closes = df['close'].astype(float).tolist()
    highs, lows = smc_find_swings(closes, window=window)
    side = smc_determine_side_from_swings(highs, lows)
    if side is None:
        # fallback to MA
        arr = df['close'].astype(float)
        if len(arr) >= 20:
            ma5 = arr[-5:].mean()
            ma20 = arr[-20:].mean()
            side = "BUY" if ma5 > ma20 else "SELL"
        else:
            side = None

    entry_price = float(df['close'].iloc[-1])
    if side == "BUY":
        sl = smc_nearest_support_buy(lows) or (entry_price - 2.0)
    else:
        sl = smc_nearest_resistance_sell(highs) or (entry_price + 2.0)

    # volume stats
    vol_series = df['volume'].astype(float).replace(0, np.nan).dropna()
    recent_vol_avg = float(vol_series.tail(50).mean()) if not vol_series.empty else 0.0
    current_vol = float(df['volume'].iloc[-1]) if 'volume' in df.columns else 0.0

    info = {
        "recent_vol_avg": recent_vol_avg,
        "current_vol": current_vol,
        "highs": highs,
        "lows": lows
    }
    return side, float(sl), df, info


# -------------------- position sizing & TP calc --------------------
def calculate_lot_size(capital, risk_percent, entry_price, sl_price):
    if capital is None or risk_percent is None:
        return 0.01
    risk_dollars = capital * (risk_percent / 100.0)
    points = abs(entry_price - sl_price) * 10.0  # كل نقطة = $0.1 لكل 0.01 lot
    if points <= 0:
        return 0.01
    lot = risk_dollars / (points * 0.1)
    lot = round(lot, 2)
    return max(lot, 0.01)


def compute_tps(entry, sl, rr_list=(2.0, 3.0)):
    """حسِب TP على أساس RR ratios"""
    dist_points = abs(entry - sl) * 10.0
    # price distance per point = 0.1$ per 0.01 lot but here we convert back to price
    price_dist = abs(entry - sl)
    tps = []
    if entry > sl:
        # sell (entry > sl) -> TP below entry
        for rr in rr_list:
            tps.append(entry - price_dist * rr)
    else:
        # buy
        for rr in rr_list:
            tps.append(entry + price_dist * rr)
    return tps


# -------------------- Telegram UI --------------------
def main_keyboard():
    kb = types.ReplyKeyboardMarkup(resize_keyboard=True)
    kb.row('/scalping', '/mysettings')
    kb.row('/setcapital', '/setrisk', '/subscribe')
    kb.row('/createnewcode', '/mycode')
    return kb


@bot.message_handler(commands=['start'])
def cmd_start(message):
    ensure_user(message.from_user.id)
    u = get_user(message.from_user.id)
    txt = "👋 أهلاً! "
    if u["capital"] and u["risk_percent"]:
        txt += f"\n💵 رأس المال: {u['capital']}$\n⚖️ نسبة المخاطرة: {u['risk_percent']}%."
        if u["subscription_end"]:
            try:
                end_dt = datetime.fromisoformat(u["subscription_end"])
                if end_dt.date() >= datetime.now(timezone.utc).date():
                    txt += f"\n✅ اشتراك نشط حتى {u['subscription_end']}."
                else:
                    txt += f"\n⚠️ انتهى اشتراكك ({u['subscription_end']}). استخدم /subscribe للتجديد."
            except Exception:
                txt += "\n🔔 حالة الاشتراك غير معروفة."
        else:
            txt += "\n🔔 أنت غير مشترك — للاشتراك استخدم /subscribe"
    else:
        txt += "\nالرجاء إعداد رأس المال ونسبة المخاطرة: /setcapital و /setrisk"

    bot.send_message(message.chat.id, txt, reply_markup=main_keyboard())


# setcapital
@bot.message_handler(commands=['setcapital'])
def cmd_setcapital(message):
    ensure_user(message.from_user.id)
    msg = bot.send_message(message.chat.id, "💰 اكتب رأس المال بالدولار (مثال: 50 أو 100.5):")
    bot.register_next_step_handler(msg, proc_setcapital)


def proc_setcapital(message):
    try:
        txt = message.text.strip().replace(',', '.')
        val = float(txt)
        if val <= 0:
            bot.send_message(message.chat.id, "❌ اكتب رقم موجب أكبر من صفر.")
            return
        set_capital(message.from_user.id, val)
        bot.send_message(message.chat.id, f"✅ تم حفظ رأس المال: {val}$.")
    except Exception:
        bot.send_message(message.chat.id, "❌ قيمة غير صحيحة. حاول مرّة أخرى.")


# setrisk
@bot.message_handler(commands=['setrisk'])
def cmd_setrisk(message):
    ensure_user(message.from_user.id)
    msg = bot.send_message(message.chat.id, "📊 اكتب نسبة المخاطرة لكل صفقة (%) مثال: 1 أو 2:")
    bot.register_next_step_handler(msg, proc_setrisk)


def proc_setrisk(message):
    try:
        txt = message.text.strip().replace(',', '.')
        val = float(txt)
        if val <= 0 or val > 100:
            bot.send_message(message.chat.id, "❌ ادخل نسبة صالحة بين 0.01 و 100.")
            return
        set_risk(message.from_user.id, val)
        bot.send_message(message.chat.id, f"✅ تم حفظ نسبة المخاطرة: {val}%.")
    except Exception:
        bot.send_message(message.chat.id, "❌ قيمة غير صحيحة. حاول مرّة أخرى.")


@bot.message_handler(commands=['mysettings'])
def cmd_mysettings(message):
    ensure_user(message.from_user.id)
    u = get_user(message.from_user.id)
    txt = f"إعداداتك:\nرأس المال: {u['capital'] or 'غير محدد'}$\nنسبة المخاطرة: {u['risk_percent'] or 'غير محددة'}%\n"
    txt += f"رصيد الكاش باك (إن وُجد): {u['creator_balance']:.2f}$\n"
    if u["subscription_end"]:
        txt += f"انتهاء الاشتراك: {u['subscription_end']}\n"
    bot.send_message(message.chat.id, txt)


# subscribe (manual confirm)
@bot.message_handler(commands=['subscribe'])
def cmd_subscribe(message):
    ensure_user(message.from_user.id)
    u = get_user(message.from_user.id)
    if not u["capital"] or not u["risk_percent"]:
        bot.send_message(message.chat.id, "⚠️ قبل الاشتراك حدّد رأس المال ونسبة المخاطرة باستخدام /setcapital و /setrisk")
        return

    kb = types.InlineKeyboardMarkup()
    kb.add(types.InlineKeyboardButton("🟢 لدي كود خصم", callback_data=f"sub_code_{message.from_user.id}"))
    kb.add(types.InlineKeyboardButton("✅ أكملت الدفع (بدون كود)", callback_data=f"sub_nocode_{message.from_user.id}"))
    bot.send_message(message.chat.id,
                     f"💳 الاشتراك الشهري: {SUBSCRIPTION_PRICE:.2f}$\nأرسل المبلغ إلى المحفظة:\n`{WALLET_ADDRESS}`\n\n"
                     "بعد التحويل اضغط أحد الأزرار:\n- إذا لديك كود خصم اختر '🟢 لدي كود خصم' ثم أدخل الكود.\n- إن لم يكن لديك كود اضغط '✅ أكملت الدفع (بدون كود)'.",
                     parse_mode="Markdown", reply_markup=kb)


@bot.callback_query_handler(func=lambda c: c.data and c.data.startswith("sub_code_"))
def sub_code_pressed(call):
    u_id = int(call.data.split("_")[-1])
    if call.from_user.id != u_id:
        bot.answer_callback_query(call.id, "هذا الزر خاص بالمستخدم الذي بدأ العملية.")
        return
    msg = bot.send_message(call.message.chat.id, "📌 أرسل كود صانع المحتوى الآن:")
    bot.register_next_step_handler(msg, proc_sub_with_code)


def proc_sub_with_code(message):
    code_txt = message.text.strip()
    ensure_user(message.from_user.id)
    code_row = get_creator_code(code_txt)
    if not code_row:
        bot.send_message(message.chat.id, "❌ الكود غير موجود.")
        return
    code, discount, creator_id = code_row
    discount = float(discount)
    amount_due = SUBSCRIPTION_PRICE * (1 - discount / 100.0)
    pid = create_subscription_payment(message.from_user.id, amount_due, code)
    bot.send_message(message.chat.id,
                     f"✅ تم تسجيل تأكيد الدفع (معلّق). المبلغ بعد الخصم: {amount_due:.2f}$.\nأرسل المبلغ إلى المحفظة ثم انتظر موافقة الأدمن.",
                     parse_mode="Markdown")
    kb = types.InlineKeyboardMarkup()
    kb.add(types.InlineKeyboardButton("✅ موافقة", callback_data=f"approve_pay_{pid}"),
           types.InlineKeyboardButton("❌ رفض", callback_data=f"reject_pay_{pid}"))
    bot.send_message(ADMIN_ID,
                     f"🟡 طلب اشتراك جديد (مستخدم {message.from_user.id})\nالمبلغ: {amount_due:.2f}$\nكود: {code}\npayment_id: {pid}",
                     reply_markup=kb)


@bot.callback_query_handler(func=lambda c: c.data and c.data.startswith("sub_nocode_"))
def sub_nocode_pressed(call):
    u_id = int(call.data.split("_")[-1])
    if call.from_user.id != u_id:
        bot.answer_callback_query(call.id, "هذا الزر خاص بالمستخدم الذي بدأ العملية.")
        return
    amount_due = SUBSCRIPTION_PRICE
    pid = create_subscription_payment(call.from_user.id, amount_due, None)
    bot.answer_callback_query(call.id, "تم تسجيل الدفع المسبق (معلق). سيتم إبلاغ الأدمن بمراجعة الدفع.")
    kb = types.InlineKeyboardMarkup()
    kb.add(types.InlineKeyboardButton("✅ موافقة", callback_data=f"approve_pay_{pid}"),
           types.InlineKeyboardButton("❌ رفض", callback_data=f"reject_pay_{pid}"))
    bot.send_message(ADMIN_ID,
                     f"🟡 طلب اشتراك جديد (مستخدم {call.from_user.id})\nالمبلغ: {amount_due:.2f}$\npayment_id: {pid}",
                     reply_markup=kb)
    bot.send_message(call.message.chat.id, f"✅ تم تسجيل أنك دفعت {amount_due:.2f}$ (معلق انتظار موافقة الأدمن).")


@bot.callback_query_handler(func=lambda c: c.data and (c.data.startswith("approve_pay_") or c.data.startswith("reject_pay_")))
def handle_payment_approval(call):
    parts = call.data.split("_")
    action = parts[0]
    payment_id = int(parts[-1])
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("SELECT user_id, amount, code FROM subscription_payments WHERE id=?", (payment_id,))
        row = c.fetchone()
    if not row:
        bot.answer_callback_query(call.id, "معرف الدفع غير موجود.")
        return
    user_id, amount, code = row
    if call.from_user.id != ADMIN_ID:
        bot.answer_callback_query(call.id, "غير مصرح لك.")
        return
    if action == "approve":
        set_sub_payment_status(payment_id, "approved")
        end = set_subscription_for_user(user_id)
        bot.send_message(user_id, f"✅ تم تفعيل اشتراكك حتى {end}. شكراً لك!")
        bot.answer_callback_query(call.id, "تمت الموافقة وتفعيل الاشتراك.")
        if code:
            add_code_usage(user_id, code, purchased=1, amount=amount)
            code_row = get_creator_code(code)
            if code_row:
                _, _, creator_id = code_row
                credit_creator(creator_id, CASHBACK_PER_PURCHASE)
                bot.send_message(creator_id, f"🔔 تم تسجيل بيع باستخدام كودك {code}. رصيدك زاد بمقدار {CASHBACK_PER_PURCHASE:.2f}$")
    else:
        set_sub_payment_status(payment_id, "rejected")
        bot.send_message(user_id, "❌ تم رفض طلب اشتراكك. يرجى التواصل مع الدعم.")
        bot.answer_callback_query(call.id, "تم رفض الطلب.")


# createnewcode / mycode / withdraw (مثل السابق، مختصر لعدم التكرار)
@bot.message_handler(commands=['createnewcode'])
def cmd_createnewcode(message):
    if message.from_user.id != ADMIN_ID:
        bot.reply_to(message, "❌ غير مصرح لك.")
        return
    msg = bot.send_message(message.chat.id, "📌 أرسل اسم الكود (مثال: mk10):")
    bot.register_next_step_handler(msg, proc_new_code_step)


def proc_new_code_step(message):
    code = message.text.strip()
    msg = bot.send_message(message.chat.id, "📊 أرسل نسبة الخصم (مثال: 30):")
    bot.register_next_step_handler(msg, lambda m: proc_new_code_discount(m, code))


def proc_new_code_discount(message, code):
    try:
        discount = float(message.text.strip().replace(',', '.'))
    except Exception:
        bot.send_message(message.chat.id, "❌ نسبة غير صحيحة.")
        return
    msg = bot.send_message(message.chat.id, "👤 أرسل أيدي صانع المحتوى (مثال: 123456789):")
    bot.register_next_step_handler(msg, lambda m: proc_new_code_save(m, code, discount))


def proc_new_code_save(message, code, discount):
    try:
        creator_id = int(message.text.strip())
    except Exception:
        bot.send_message(message.chat.id, "❌ اكتب رقم ايدي صحيح.")
        return
    create_creator_code(code, discount, creator_id)
    bot.send_message(message.chat.id, f"✅ تم إنشاء الكود `{code}` بالخصم {discount}% لصاحب الايدي {creator_id}.", parse_mode="Markdown")


@bot.message_handler(commands=['mycode'])
def cmd_mycode(message):
    ensure_user(message.from_user.id)
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("SELECT code, discount_percent FROM creators_codes WHERE creator_id=?", (message.from_user.id,))
        row = c.fetchone()
        if not row:
            bot.send_message(message.chat.id, "❌ ليس لديك كود مسجل.")
            return
        code, discount = row
        c.execute("SELECT COUNT(DISTINCT user_id), SUM(purchased), SUM(amount) FROM code_usage WHERE code=?", (code,))
        stats = c.fetchone()
        users_count = stats[0] or 0
        purchasers = stats[1] or 0
        total_amount = (stats[2] or 0.0)
        c.execute("SELECT creator_balance FROM users WHERE user_id=?", (message.from_user.id,))
        bal_row = c.fetchone()
        balance = bal_row[0] if bal_row else 0.0
    bot.send_message(message.chat.id,
                     f"🎯 كودك: {code}\n💰 خصم: {discount}%\n👥 مستخدمون: {users_count}\n🛒 مشترون: {purchasers}\n💵 إجمالي: {total_amount:.2f}$\n💸 رصيدك للسحب: {balance:.2f}$")


@bot.message_handler(commands=['withdrawrequests'])
def withdraw_requests_admin(message):
    if message.from_user.id != ADMIN_ID:
        bot.send_message(message.chat.id, "❌ غير مصرح لك.")
        return
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("SELECT id, creator_id, amount, wallet_address FROM withdraw_requests WHERE status='pending'")
        rows = c.fetchall()
        if not rows:
            bot.send_message(message.chat.id, "📭 لا يوجد طلبات معلقة.")
            return
        for r in rows:
            req_id, creator_id, amount, wallet = r
            kb = types.InlineKeyboardMarkup()
            kb.add(types.InlineKeyboardButton("✅ موافقة", callback_data=f"approve_withdraw_{req_id}"),
                   types.InlineKeyboardButton("❌ رفض", callback_data=f"reject_withdraw_{req_id}"))
            bot.send_message(message.chat.id, f"طلب #{req_id}\n👤 {creator_id}\n💵 {amount}$\n🏦 {wallet}", reply_markup=kb)


@bot.callback_query_handler(func=lambda c: c.data and (c.data.startswith("approve_withdraw_") or c.data.startswith("reject_withdraw_")))
def handle_withdraw_approval(call):
    parts = call.data.split("_")
    action = parts[0]
    req_id = int(parts[-1])
    if call.from_user.id != ADMIN_ID:
        bot.answer_callback_query(call.id, "غير مصرح لك.")
        return
    if action == "approve":
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("SELECT creator_id, amount, wallet_address FROM withdraw_requests WHERE id=?", (req_id,))
            row = c.fetchone()
            if not row:
                bot.answer_callback_query(call.id, "الطلب غير موجود.")
                return
            creator_id, amount, wallet = row
            c.execute("UPDATE withdraw_requests SET status='approved' WHERE id=?", (req_id,))
            c.execute("UPDATE users SET creator_balance = creator_balance - ? WHERE user_id=?", (amount, creator_id))
            conn.commit()
        bot.send_message(creator_id, f"✅ تم الموافقة على سحب {amount:.2f}$ إلى المحفظة: {wallet}. تحقق من محفظتك.")
        bot.answer_callback_query(call.id, "تم الموافقة.")
    else:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("UPDATE withdraw_requests SET status='rejected' WHERE id=?", (req_id,))
            conn.commit()
        bot.answer_callback_query(call.id, "تم الرفض.")
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("SELECT creator_id FROM withdraw_requests WHERE id=?", (req_id,))
            r = c.fetchone()
            if r:
                bot.send_message(r[0], f"❌ طلب السحب #{req_id} تم رفضه من الأدمن.")


# -------------------- /scalping (multi-timeframe consensus + volume filter + TP) --------------------
@bot.message_handler(commands=['scalping'])
def cmd_scalping(message):
    ensure_user(message.from_user.id)
    u = get_user(message.from_user.id)

    # subscription check
    if not u["subscription_end"]:
        bot.send_message(message.chat.id, "🚫 يجب الاشتراك الشهري أولاً عبر /subscribe")
        return
    try:
        sub_end = datetime.fromisoformat(u["subscription_end"])
    except Exception:
        bot.send_message(message.chat.id, "⚠️ خطأ في حالة الاشتراك، تواصل مع الأدمن.")
        return
    if sub_end.date() < datetime.now(timezone.utc).date():
        bot.send_message(message.chat.id, "🔒 اشتراكك انتهى — جدد عبر /subscribe")
        return

    if tv is None:
        bot.send_message(message.chat.id, "⚠️ tvDatafeed غير مفعّل هنا. لا أستطيع جلب بيانات.")
        return

    bot.send_message(message.chat.id, "⏳ جاري تحليل السوق (SMC) على فريمات: 5m, 30m, 1h, 4h ...")

    tfs = [
        ("5m", TvdfInterval.in_5_minute, 300),
        ("30m", TvdfInterval.in_30_minute, 300),
        ("1h", TvdfInterval.in_1_hour, 300),
        ("4h", TvdfInterval.in_4_hour, 300)
    ]

    results = []
    errors = []
    for name, interval, bars in tfs:
        side, sl, df, info = analyze_tf(SYMBOL, EXCHANGE, interval, bars=bars, window=5, message=message)
        if side is None:
            errors.append(name)
            continue
        results.append((name, side, sl, df, info))

    if errors and len(errors) == len(tfs):
        bot.send_message(message.chat.id, "⚠️ فشل التحليل على كل الفريمات (بيانات غير كافية).")
        return

    # if some frames failed, still proceed with available ones
    votes = {"BUY": 0, "SELL": 0}
    for (_, side, _, _, _) in results:
        votes[side] += 1

    chosen = None
    for s in votes:
        if votes[s] >= 3:  # إجماع 3 من 4
            chosen = s
            break

    if chosen is None:
        # no strong consensus -> return message showing votes
        bot.send_message(message.chat.id, f"⚠️ لا إجماع قوي بين الفريمات. الأصوات: BUY={votes['BUY']}, SELL={votes['SELL']}.")
        return

    # pick 30m as entry price reference if available
    entry_price = None
    sl_price = None
    vol_warnings = []
    for name, side, sl, df, info in results:
        if name == "30m":
            entry_price = float(df['close'].iloc[-1])
            sl_price = sl
        # volume check: if current < 0.5 * recent_avg -> warn
        if info and info["recent_vol_avg"] > 0:
            if info["current_vol"] < 0.5 * info["recent_vol_avg"]:
                vol_warnings.append(name)

    if entry_price is None:
        # fallback to first available
        entry_price = float(results[0][3]['close'].iloc[-1])
        sl_price = results[0][2]

    # sizing & TP
    capital = u["capital"]
    risk_percent = u["risk_percent"]
    if not capital or not risk_percent:
        bot.send_message(message.chat.id, "⚠️ الرجاء تحديد رأس المال ونسبة المخاطرة أولاً (/setcapital و /setrisk).")
        return

    risk_dollars = capital * (risk_percent / 100.0)
    points = abs(entry_price - sl_price) * 10.0
    min_loss = points * 0.1  # خسارة 0.01 lot

    lot = calculate_lot_size(capital, risk_percent, entry_price, sl_price)
    tps = compute_tps(entry_price, sl_price, rr_list=(2.0, 3.0))

    # Build message
    def fmt(x): return f"{x:.2f}"
    header = f"{SYMBOL} {'BUY' if chosen=='BUY' else 'SELL'} @ {fmt(entry_price)}  (إجماع {votes[chosen]}/{len(results)})"
    body = [
        f"⚫ SL : {fmt(sl_price)}",
        f"💵 رأس المال: {capital}$",
        f"⚠️ نسبة المخاطرة: {risk_percent}% → {risk_dollars:.2f}$",
        f"📏 المسافة للستوب: {points:.1f} نقاط (كل نقطة = $0.1 لكل 0.01 لوت)",
        f"🔢 حجم اللوت المقترح: {lot} lot",
        f"🎯 TP1 (RR 1:2): {fmt(tps[0])}",
        f"🎯 TP2 (RR 1:3): {fmt(tps[1])}",
    ]
    if min_loss > risk_dollars:
        body.append(f"⚠️ تحذير: حتى بأصغر لوت ستخسر {min_loss:.2f}$ — أكبر من المسموح ({risk_dollars:.2f}$). الصفقة للمعرفة فقط.")
    if vol_warnings:
        body.append("⚠️ تحذير سيولة: أحجام التداول ضعيفة على الفريمات: " + ", ".join(vol_warnings))
    body.append("\n✅ تحليل متعدد الفريمات (SMC مبسطة + فلتر حجم). اتبع إدارة رأس المال دائماً.")
    bot.send_message(message.chat.id, header + "\n" + "\n".join(body))


# -------------------- Run --------------------
if __name__ == "__main__":
    print("✅ Bot is running...")
    try:
        bot.infinity_polling(timeout=60, long_polling_timeout=30)
    except KeyboardInterrupt:
        print("Stopped by user")
    except Exception:
        traceback.print_exc()
