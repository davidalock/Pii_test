# Full Merchant PII Dataset Summary

## Dataset Overview
- **Total Records**: 9,056 (using every record from merchant frame)
- **Unique Addresses**: 8,192 UK merchant/ATM locations
- **Total PII Data Points**: 108,672 (12 fields × 9,056 records)
- **Geographic Coverage**: UK-wide merchant and ATM locations

## Data Fields Included
1. **Personal Information**:
   - First Name (from MP list)
   - Surname (from MP list)
   - Full Name
   - Email Address (100 realistic domains)
   - Phone Number (UK format)
   - Date of Birth

2. **UK-Specific Financial Data**:
   - UK National Insurance Number (NINO)
   - PAN (Credit/Debit Card) Numbers
   - UK Bank Sort Codes
   - UK Account Numbers
   - UK IBAN Numbers

3. **Address Information**:
   - Full Address (from merchant frame)
   - Merchant Name
   - Merchant Types
   - Geographic Coordinates

## PII Detection Performance (1,000 sample test)
| Field | Detection Rate | Notes |
|-------|---------------|-------|
| Email Address | 100.0% | Perfect detection |
| Address | 100.0% | Full address recognition |
| Date of Birth | 100.0% | All formats detected |
| UK NINO | 100.0% | Custom recognizer |
| UK Sort Code | 100.0% | Custom recognizer |
| UK Account Number | 100.0% | Standard banking detection |
| UK IBAN | 100.0% | Custom recognizer |
| PAN Number | 96.8% | Credit card detection |
| Full Name | 94.3% | Person entity recognition |
| Phone Number | 88.4% | UK format detection |
| First Name | 73.9% | Individual name detection |
| Surname | 54.4% | Individual name detection |

## Entity Types Detected
- **DATE_TIME**: 19.6% (dates, times, ages)
- **US_DRIVER_LICENSE**: 12.7% (false positives from numbers)
- **PERSON**: 12.2% (names, people)
- **LOCATION**: 8.6% (addresses, places)
- **URL**: 6.1% (email domains)
- **EMAIL_ADDRESS**: 5.0% (email detection)
- **UK_NINO**: 5.0% (custom recognizer)
- **UK_SORT_CODE**: 5.0% (custom recognizer)
- **UK_IBAN**: 5.0% (custom recognizer)
- **UK_POSTCODE**: 5.0% (custom recognizer)

## Key Features
- **Realistic Names**: 651 UK MPs providing diverse name combinations
- **Domain Diversity**: 100 domains (30 personal + 70 business)
- **UK Financial Standards**: Proper NINO, sort code, IBAN formats
- **Geographic Distribution**: Real UK merchant locations
- **Card Types**: Visa, Mastercard, American Express formats

## Files Generated
- `full_merchant_pii_data.csv`: Complete dataset with all 9,056 records
- `create_full_merchant_pii_data.py`: Generation script
- `test_full_merchant_pii.py`: PII detection testing script

This dataset provides comprehensive coverage for testing PII detection systems with realistic UK-specific personal and financial information at scale.
