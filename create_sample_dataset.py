import pandas as pd

# Sample spam and ham emails for testing
sample_data = {
    'v1': ['spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham',
           'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham'] * 10,
    'v2': [
        'WINNER!! You have won a $1000 cash prize! Call now to claim your reward!',
        'Hey, are you free for lunch tomorrow? Let me know.',
        'Congratulations! You have been selected for a free vacation. Click here now!',
        'Thanks for sending the report. I will review it and get back to you.',
        'URGENT: Your account has been compromised. Verify your identity immediately!',
        'Can you send me the meeting notes from yesterday? Thanks!',
        'Get rich quick! Make $5000 per week working from home. Limited time offer!',
        'I received your email. Let me check my calendar and confirm the appointment.',
        'FREE iPhone! You are the lucky winner. Claim your prize before it expires!',
        'Please find attached the document you requested. Let me know if you need anything else.',
        'Act now! Limited time offer. Buy one get one free. Click here to order!',
        'Hi, I hope you are doing well. Just wanted to follow up on our last conversation.',
        'You have won a lottery! Send your bank details to claim $10,000 prize money!',
        'The project deadline has been extended to next Friday. Please plan accordingly.',
        'SPECIAL OFFER: 90% discount on all products. Shop now before stock runs out!',
        'Could you please review the attached proposal and share your feedback?',
        'Earn money online! No experience needed. Start making $100 per hour today!',
        'Thank you for your interest. We will get back to you within 2 business days.',
        'ALERT: Suspicious activity detected on your account. Click to secure your account!',
        'I have scheduled the meeting for next Monday at 10 AM. See you there.'
    ] * 10
}

df = pd.DataFrame(sample_data)
df.to_csv('d:\\spam detection\\spam.csv', index=False, encoding='latin-1')
print(f"Sample dataset created: {len(df)} emails")
print(f"Spam: {(df['v1'] == 'spam').sum()}, Ham: {(df['v1'] == 'ham').sum()}")
