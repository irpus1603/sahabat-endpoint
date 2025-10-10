import smtplib
#from email.mime.multipart import MIMEMultipart
#from email.mime.text import MIMEText
import requests
import cv2
import os
import logging
from django.conf import settings
from core.models import SystemConfig


"""def send_email_notification(recipient_email, subject, message):
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    sender_email = "scctv5490@gmail.com"
    sender_password = "watc hior xmgg gace"
    
    try:
        # Set up the MIME
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = subject

        # Attach the message body
        msg.attach(MIMEText(message, 'plain'))

        # Create the SMTP session
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()  # Secure the connection
        server.login(sender_email, sender_password)  # Login

        # Send the email
        server.sendmail(sender_email, recipient_email, msg.as_string())
        server.quit()
        print(f"Email sent successfully to: {recipient_email}")

    except Exception as e:
        print(f"Failed to send email. Error: {str(e)}")
"""

def send_telegram_notification(frame, message, count, cameraname, current_time):
    
    # Get multiple chat IDs from config (format: id1, id2, id3)
    chat_ids_str = SystemConfig.get_value('telegram_chat_id')
    bot_token = SystemConfig.get_value('telegram_bot_token')
    
    if not chat_ids_str or not bot_token:
        print("Missing Telegram configuration (chat_id or bot_token)")
        return

    # Parse multiple chat IDs
    chat_ids = []
    try:
        # Split by comma and clean up each ID
        for chat_id in chat_ids_str.split(','):
            chat_id = chat_id.strip()
            if chat_id:
                chat_ids.append(chat_id)
    except Exception as e:
        print(f"Error parsing chat IDs: {e}")
        return
    
    if not chat_ids:
        print("No valid chat IDs found")
        return

    # Prepare message
    message_body = message
    message_body = message_body.replace("<count>", str(count))
    message_body = message_body.replace("<cameraname>", cameraname)
    message_body = message_body.replace("<time>", str(current_time))

    # Prepare image file
    telegram_url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
    timestamp = current_time
    snapshot_filename = f"{cameraname} - {timestamp}.jpg"
    snapshot_path = os.path.join(settings.MEDIA_ROOT, 'Snapshot/', snapshot_filename)

    os.makedirs(os.path.dirname(snapshot_path), exist_ok=True)
    cv2.imwrite(snapshot_path, frame)

    # Send to each chat ID
    success_count = 0
    failed_count = 0
    
    for chat_id in chat_ids:
        try:
            with open(snapshot_path, 'rb') as photo_file:
                files = {'photo': photo_file}
                data = {
                    'chat_id': chat_id,
                    'caption': message_body,
                    'parse_mode': 'HTML',
                }
                
                response = requests.post(telegram_url, files=files, data=data)
                
                if response.status_code == 200:
                    success_count += 1
                    print(f"Notification sent successfully to chat_id: {chat_id}")
                else:
                    failed_count += 1
                    print(f"Failed to send to chat_id {chat_id}. Status: {response.status_code}")
                    print(f"Response: {response.text}")
                    
        except requests.exceptions.RequestException as e:
            failed_count += 1
            print(f"Network error sending to chat_id {chat_id}: {e}")
        except Exception as e:
            failed_count += 1
            print(f"Error sending to chat_id {chat_id}: {e}")
    
    print(f"Telegram notification summary: {success_count} successful, {failed_count} failed")
    
    # Clean up snapshot file
    try:
        if os.path.exists(snapshot_path):
            os.remove(snapshot_path)
    except Exception as e:
        print(f"Error cleaning up snapshot file: {e}")


