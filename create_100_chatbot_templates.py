#!/usr/bin/env python3
"""
Script to generate 100 chat input templates focused on realistic chatbot conversation styles.
These templates cover various scenarios users might encounter when interacting with chatbots:
- Customer service inquiries
- Account management
- Product support
- General assistance
- Personal information sharing
- Business inquiries
"""

import pandas as pd
import csv
import random

def create_chatbot_templates():
    """Create 100 diverse chat templates focused on realistic chatbot interactions"""
    
    # Read the merchant frame data to get location information
    merchant_file = '/Users/davidlock/Downloads/soccer data python/testing poe/merchant_frame for ATM.csv'
    df = pd.read_csv(merchant_file)
    
    # Get unique locations from the formatted_address column
    locations = df['formatted_address'].dropna().tolist()
    
    # Read the MP list data to get forename, surname, and email information
    mp_file = '/Users/davidlock/Downloads/soccer data python/testing poe/mplist.csv'
    mp_df = pd.read_csv(mp_file)
    
    # Get the personal data from MP list
    forenames = mp_df['Forename'].dropna().tolist()
    surnames = mp_df['Surname'].dropna().tolist()
    emails = mp_df['Email'].dropna().tolist()
    
    # Create 100 chatbot-focused templates organized by categories
    templates = [
        
        # === CUSTOMER SERVICE & SUPPORT (20 templates) ===
        "Hi, I need help with my account. My name is &&forename&& &&surname&& and my email is &&email&&. I'm calling from &&location&&.",
        
        "Hello, I'm having an issue with my order. I'm &&forename&& &&surname&&, email &&email&&, shipping address is &&location&&.",
        
        "Good morning, I need to update my profile. My current name is &&forename&& &&surname&&, email &&email&&, and I live at &&location&&.",
        
        "Hi there! I'm &&forename&& &&surname&& and I need assistance. You can reach me at &&email&& or find me at &&location&&.",
        
        "Hello, I'm experiencing a technical problem. My details: &&forename&& &&surname&&, &&email&&, located at &&location&&.",
        
        "Hi, I need to report a billing issue. I'm &&forename&& &&surname&&, my contact email is &&email&&, billing address: &&location&&.",
        
        "Good afternoon, I have a question about my subscription. Name: &&forename&& &&surname&&, email: &&email&&, address: &&location&&.",
        
        "Hello, I need help resetting my password. I'm &&forename&& &&surname&&, registered email: &&email&&, from &&location&&.",
        
        "Hi, I want to cancel my service. My account details: &&forename&& &&surname&&, &&email&&, service address: &&location&&.",
        
        "Good morning, I need to change my delivery address. I'm &&forename&& &&surname&&, email &&email&&, new address: &&location&&.",
        
        "Hello, I'm locked out of my account. User: &&forename&& &&surname&&, email: &&email&&, registered from &&location&&.",
        
        "Hi, I need a refund for my recent purchase. Customer: &&forename&& &&surname&&, contact: &&email&&, billing address: &&location&&.",
        
        "Good day, I have a complaint to file. My name is &&forename&& &&surname&&, email &&email&&, incident location: &&location&&.",
        
        "Hello, I need technical support urgently. I'm &&forename&& &&surname&&, you can reach me at &&email&&, I'm based at &&location&&.",
        
        "Hi, I want to upgrade my plan. Account holder: &&forename&& &&surname&&, email: &&email&&, service location: &&location&&.",
        
        "Good morning, I need help with installation. Customer: &&forename&& &&surname&&, contact: &&email&&, install address: &&location&&.",
        
        "Hello, I'm having trouble logging in. My credentials: &&forename&& &&surname&&, &&email&&, account registered to &&location&&.",
        
        "Hi, I need to update my payment method. Name: &&forename&& &&surname&&, email: &&email&&, billing address: &&location&&.",
        
        "Good afternoon, I have feedback about your service. I'm &&forename&& &&surname&&, email &&email&&, service area: &&location&&.",
        
        "Hello, I need help with app functionality. User: &&forename&& &&surname&&, contact: &&email&&, using from &&location&&.",
        
        # === ACCOUNT MANAGEMENT (15 templates) ===
        "I'd like to create a new account. My details are: &&forename&& &&surname&&, &&email&&, and I'm located at &&location&&.",
        
        "Please update my account information. Current: &&forename&& &&surname&&, email: &&email&&, address: &&location&&.",
        
        "I need to verify my account. My name is &&forename&& &&surname&&, registered email: &&email&&, from &&location&&.",
        
        "Can you help me recover my account? I'm &&forename&& &&surname&&, my email should be &&email&&, registered at &&location&&.",
        
        "I want to delete my account permanently. Account details: &&forename&& &&surname&&, &&email&&, location: &&location&&.",
        
        "Please merge my accounts. Primary: &&forename&& &&surname&&, email: &&email&&, associated with &&location&&.",
        
        "I need to transfer my account to a new email. Current user: &&forename&& &&surname&&, old email: &&email&&, address: &&location&&.",
        
        "Can you activate my account? Details: &&forename&& &&surname&&, email: &&email&&, registration address: &&location&&.",
        
        "I want to suspend my account temporarily. User: &&forename&& &&surname&&, contact: &&email&&, account location: &&location&&.",
        
        "Please reactivate my suspended account. I'm &&forename&& &&surname&&, email: &&email&&, originally from &&location&&.",
        
        "I need to change my username. Current account: &&forename&& &&surname&&, email: &&email&&, registered at &&location&&.",
        
        "Can you update my security settings? Account holder: &&forename&& &&surname&&, &&email&&, location: &&location&&.",
        
        "I want to add two-factor authentication. My details: &&forename&& &&surname&&, email: &&email&&, based at &&location&&.",
        
        "Please help me set up my profile. New user: &&forename&& &&surname&&, contact: &&email&&, location: &&location&&.",
        
        "I need to link multiple accounts. Primary user: &&forename&& &&surname&&, main email: &&email&&, address: &&location&&.",
        
        # === BOOKING & APPOINTMENTS (15 templates) ===
        "I'd like to book an appointment. My name is &&forename&& &&surname&&, contact email: &&email&&, I'm located at &&location&&.",
        
        "Can I schedule a consultation? I'm &&forename&& &&surname&&, you can reach me at &&email&&, address: &&location&&.",
        
        "I need to reschedule my appointment. Customer: &&forename&& &&surname&&, email: &&email&&, appointment location: &&location&&.",
        
        "Please cancel my booking. Details: &&forename&& &&surname&&, contact: &&email&&, originally booked for &&location&&.",
        
        "I want to book a table for tonight. Name: &&forename&& &&surname&&, email: &&email&&, party location preference: &&location&&.",
        
        "Can I reserve a service? Customer details: &&forename&& &&surname&&, &&email&&, service needed at &&location&&.",
        
        "I'd like to book a delivery slot. My details: &&forename&& &&surname&&, email: &&email&&, delivery address: &&location&&.",
        
        "Please schedule a pickup. Customer: &&forename&& &&surname&&, contact: &&email&&, pickup location: &&location&&.",
        
        "I need to book urgent service. Name: &&forename&& &&surname&&, email: &&email&&, service address: &&location&&.",
        
        "Can I schedule a follow-up? Previous customer: &&forename&& &&surname&&, &&email&&, service location: &&location&&.",
        
        "I want to book a recurring service. Details: &&forename&& &&surname&&, contact: &&email&&, service address: &&location&&.",
        
        "Please help me book group service. Organizer: &&forename&& &&surname&&, email: &&email&&, venue: &&location&&.",
        
        "I need to modify my reservation. Customer: &&forename&& &&surname&&, contact: &&email&&, current booking: &&location&&.",
        
        "Can I get a booking confirmation? Name: &&forename&& &&surname&&, email: &&email&&, booking location: &&location&&.",
        
        "I'd like to check availability. Potential customer: &&forename&& &&surname&&, &&email&&, interested in &&location&&.",
        
        # === PRODUCT INQUIRIES (15 templates) ===
        "I'm interested in your product. My name is &&forename&& &&surname&&, email: &&email&&, I'm shopping from &&location&&.",
        
        "Can you tell me more about pricing? Customer: &&forename&& &&surname&&, contact: &&email&&, shipping to &&location&&.",
        
        "I need product specifications. Inquiry from: &&forename&& &&surname&&, &&email&&, project location: &&location&&.",
        
        "Do you have this item in stock? Customer: &&forename&& &&surname&&, email: &&email&&, store preference: &&location&&.",
        
        "I want to compare products. Shopper: &&forename&& &&surname&&, contact: &&email&&, delivery area: &&location&&.",
        
        "Can I get a product demo? Interested party: &&forename&& &&surname&&, &&email&&, demo location: &&location&&.",
        
        "I need bulk pricing information. Business contact: &&forename&& &&surname&&, email: &&email&&, business address: &&location&&.",
        
        "Is there a warranty on this product? Customer: &&forename&& &&surname&&, contact: &&email&&, purchase location: &&location&&.",
        
        "I have questions about compatibility. User: &&forename&& &&surname&&, email: &&email&&, installation site: &&location&&.",
        
        "Can you recommend alternatives? Shopper: &&forename&& &&surname&&, &&email&&, shopping from &&location&&.",
        
        "I need installation support. Customer: &&forename&& &&surname&&, contact: &&email&&, install address: &&location&&.",
        
        "What's included in this package? Inquiry: &&forename&& &&surname&&, email: &&email&&, delivery to &&location&&.",
        
        "Can I customize this product? Customer: &&forename&& &&surname&&, &&email&&, project location: &&location&&.",
        
        "I need technical documentation. Requester: &&forename&& &&surname&&, contact: &&email&&, implementation at &&location&&.",
        
        "Is there educational pricing? Student: &&forename&& &&surname&&, email: &&email&&, institution: &&location&&.",
        
        # === FEEDBACK & REVIEWS (10 templates) ===
        "I'd like to leave feedback. Customer: &&forename&& &&surname&&, email: &&email&&, service received at &&location&&.",
        
        "I want to write a review. Previous customer: &&forename&& &&surname&&, contact: &&email&&, experience at &&location&&.",
        
        "I have a suggestion for improvement. User: &&forename&& &&surname&&, &&email&&, feedback from &&location&&.",
        
        "I'd like to report a positive experience. Happy customer: &&forename&& &&surname&&, email: &&email&&, service location: &&location&&.",
        
        "I want to share my success story. Satisfied client: &&forename&& &&surname&&, contact: &&email&&, project at &&location&&.",
        
        "I have constructive criticism. Customer: &&forename&& &&surname&&, &&email&&, experience at &&location&&.",
        
        "I'd like to recommend your service. Advocate: &&forename&& &&surname&&, email: &&email&&, referring from &&location&&.",
        
        "I want to participate in your survey. Participant: &&forename&& &&surname&&, contact: &&email&&, responding from &&location&&.",
        
        "I have ideas for new features. User: &&forename&& &&surname&&, &&email&&, using from &&location&&.",
        
        "I'd like to join your beta program. Volunteer: &&forename&& &&surname&&, contact: &&email&&, testing from &&location&&.",
        
        # === PERSONAL INTRODUCTIONS (10 templates) ===
        "Hi! I'm new here. My name is &&forename&& &&surname&&, email: &&email&&, just moved to &&location&&.",
        
        "Hello everyone! I'm &&forename&& &&surname&&, you can reach me at &&email&&, I'm based in &&location&&.",
        
        "Greetings! I'm &&forename&& &&surname&&, my contact email is &&email&&, and I work from &&location&&.",
        
        "Nice to meet you all! I'm &&forename&& &&surname&&, email &&email&&, originally from &&location&&.",
        
        "Hi there! I'm &&forename&& &&surname&&, feel free to email me at &&email&&, I live near &&location&&.",
        
        "Good morning! I'm &&forename&& &&surname&&, my email address is &&email&&, and I'm visiting &&location&&.",
        
        "Hello! I'm &&forename&& &&surname&&, you can contact me via &&email&&, currently staying at &&location&&.",
        
        "Hey everyone! I'm &&forename&& &&surname&&, drop me a line at &&email&&, I'm working in &&location&&.",
        
        "Pleased to meet you! I'm &&forename&& &&surname&&, my email is &&email&&, and I represent &&location&&.",
        
        "Good day! I'm &&forename&& &&surname&&, contact email: &&email&&, and I'm traveling through &&location&&.",
        
        # === BUSINESS INQUIRIES (15 templates) ===
        "I'm interested in your services for my business. Contact: &&forename&& &&surname&&, email: &&email&&, business location: &&location&&.",
        
        "Can we discuss a partnership? Business owner: &&forename&& &&surname&&, &&email&&, company based at &&location&&.",
        
        "I'd like a quote for corporate services. Decision maker: &&forename&& &&surname&&, contact: &&email&&, office at &&location&&.",
        
        "We need enterprise solutions. Representative: &&forename&& &&surname&&, email: &&email&&, headquarters: &&location&&.",
        
        "I'm looking for B2B pricing. Procurement: &&forename&& &&surname&&, &&email&&, purchasing for &&location&&.",
        
        "Can you provide references? Potential client: &&forename&& &&surname&&, contact: &&email&&, evaluating for &&location&&.",
        
        "I need a service level agreement. Business contact: &&forename&& &&surname&&, email: &&email&&, site: &&location&&.",
        
        "We're interested in white-label solutions. Partner: &&forename&& &&surname&&, &&email&&, operations at &&location&&.",
        
        "I'd like to discuss volume discounts. Buyer: &&forename&& &&surname&&, contact: &&email&&, purchasing for &&location&&.",
        
        "Can we schedule a business consultation? Executive: &&forename&& &&surname&&, email: &&email&&, office location: &&location&&.",
        
        "I need integration support for our platform. Technical lead: &&forename&& &&surname&&, &&email&&, development at &&location&&.",
        
        "We're evaluating vendors. Evaluator: &&forename&& &&surname&&, contact: &&email&&, company at &&location&&.",
        
        "I want to discuss licensing terms. Legal contact: &&forename&& &&surname&&, email: &&email&&, jurisdiction: &&location&&.",
        
        "Can you provide training for our team? Manager: &&forename&& &&surname&&, &&email&&, team located at &&location&&.",
        
        "I'm interested in reseller opportunities. Sales prospect: &&forename&& &&surname&&, contact: &&email&&, territory: &&location&&."
    ]
    
    # Define template categories for analysis
    categories = {
        'Customer Service & Support': list(range(0, 20)),
        'Account Management': list(range(20, 35)),
        'Booking & Appointments': list(range(35, 50)),
        'Product Inquiries': list(range(50, 65)),
        'Feedback & Reviews': list(range(65, 75)),
        'Personal Introductions': list(range(75, 85)),
        'Business Inquiries': list(range(85, 100))
    }
    
    return templates, categories, forenames, surnames, emails, locations

def generate_chatbot_samples(num_samples=1000):
    """Generate sample chat inputs using the 100 chatbot templates"""
    
    templates, categories, forenames, surnames, emails, locations = create_chatbot_templates()
    
    samples = []
    category_counts = {cat: 0 for cat in categories.keys()}
    
    for i in range(num_samples):
        # Select random template
        template_idx = random.randint(0, len(templates) - 1)
        template = templates[template_idx]
        
        # Determine category
        template_category = None
        for cat_name, indices in categories.items():
            if template_idx in indices:
                template_category = cat_name
                category_counts[cat_name] += 1
                break
        
        # Replace placeholders with random data
        filled_template = template
        filled_template = filled_template.replace('&&forename&&', random.choice(forenames))
        filled_template = filled_template.replace('&&surname&&', random.choice(surnames))
        filled_template = filled_template.replace('&&email&&', random.choice(emails))
        filled_template = filled_template.replace('&&location&&', random.choice(locations))
        
        samples.append({
            'template_id': template_idx,
            'category': template_category,
            'chat_input': filled_template
        })
    
    return samples, category_counts

def save_chatbot_templates():
    """Save the 100 chatbot templates to a CSV file"""
    
    templates, categories, _, _, _, _ = create_chatbot_templates()
    
    # Create template data with categories
    template_data = []
    for i, template in enumerate(templates):
        # Find category for this template
        category = None
        for cat_name, indices in categories.items():
            if i in indices:
                category = cat_name
                break
        
        template_data.append({
            'template_id': i,
            'category': category,
            'template': template
        })
    
    # Save to CSV
    output_file = '/Users/davidlock/Downloads/soccer data python/testing poe/chatbot_templates_100.csv'
    
    with open(output_file, 'w', newline='', encoding='utf-8') as file:
        fieldnames = ['template_id', 'category', 'template']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        
        writer.writeheader()
        for template_data_row in template_data:
            writer.writerow(template_data_row)
    
    print(f"✅ Saved 100 chatbot templates to: {output_file}")
    
    # Print category summary
    print(f"\n📊 TEMPLATE CATEGORIES:")
    for cat_name, indices in categories.items():
        print(f"   {cat_name}: {len(indices)} templates")
    
    return output_file

def demo_chatbot_samples():
    """Generate and display sample chatbot conversations"""
    
    print("=== DEMO: CHATBOT CONVERSATION SAMPLES ===\n")
    
    samples, category_counts = generate_chatbot_samples(20)  # Generate 20 demo samples
    
    for i, sample in enumerate(samples, 1):
        print(f"{i}. [{sample['category']}]")
        print(f"   {sample['chat_input']}")
        print()
    
    print("📊 CATEGORY DISTRIBUTION (in 20 samples):")
    for category, count in category_counts.items():
        print(f"   {category}: {count} samples")

if __name__ == "__main__":
    print("🤖 CREATING 100 CHATBOT-FOCUSED TEMPLATES")
    print("=" * 60)
    
    # Save templates to file
    template_file = save_chatbot_templates()
    
    # Show demo samples
    demo_chatbot_samples()
    
    print(f"\n✅ SUMMARY:")
    print(f"   📝 100 templates created across 7 categories")
    print(f"   💾 Saved to: chatbot_templates_100.csv")
    print(f"   🎯 Focused on realistic chatbot conversation styles")
    print(f"   📊 Categories: Customer Service, Account Management, Bookings, Products, Feedback, Introductions, Business")
