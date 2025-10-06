import boto3
import json
from botocore.exceptions import ClientError, NoCredentialsError
import logging

def test_aws_connectivity():
    """Comprehensive AWS connectivity testing"""
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    tests = []
    
    # Test 1: Basic AWS credentials
    try:
        sts = boto3.client('sts')
        identity = sts.get_caller_identity()
        logger.info(f"✅ AWS Identity: {identity['Arn']}")
        tests.append(("AWS Credentials", True, "Valid credentials found"))
    except NoCredentialsError:
        logger.error("❌ No AWS credentials found")
        tests.append(("AWS Credentials", False, "No credentials configured"))
        return tests
    except Exception as e:
        logger.error(f"❌ AWS credential error: {str(e)}")
        tests.append(("AWS Credentials", False, str(e)))
        return tests
    
    # Test 2: S3 Access
    try:
        s3 = boto3.client('s3')
        buckets = s3.list_buckets()
        logger.info(f"✅ S3 Access: Found {len(buckets['Buckets'])} buckets")
        tests.append(("S3 Access", True, f"Can list {len(buckets['Buckets'])} buckets"))
    except Exception as e:
        logger.error(f"❌ S3 Access failed: {str(e)}")
        tests.append(("S3 Access", False, str(e)))
    
    # Test 3: CloudWatch Access
    try:
        cloudwatch = boto3.client('cloudwatch')
        metrics = cloudwatch.list_metrics(MaxRecords=1)
        logger.info("✅ CloudWatch Access: Can list metrics")
        tests.append(("CloudWatch Access", True, "Can list metrics"))
    except Exception as e:
        logger.error(f"❌ CloudWatch Access failed: {str(e)}")
        tests.append(("CloudWatch Access", False, str(e)))
    
    # Test 4: IAM Permissions
    try:
        iam = boto3.client('iam')
        user = iam.get_user()
        logger.info(f"✅ IAM Access: Current user {user['User']['UserName']}")
        tests.append(("IAM Access", True, f"User: {user['User']['UserName']}"))
    except Exception as e:
        logger.warning(f"⚠️ IAM Access limited: {str(e)}")
        tests.append(("IAM Access", False, str(e)))
    
    # Test 5: Create test bucket (optional)
    test_bucket_name = 'black-friday-mlops-test-bucket'
    try:
        s3 = boto3.client('s3')
        s3.create_bucket(Bucket=test_bucket_name)
        logger.info(f"✅ S3 Write Access: Created test bucket {test_bucket_name}")
        
        # Clean up
        s3.delete_bucket(Bucket=test_bucket_name)
        logger.info(f"✅ S3 Delete Access: Deleted test bucket {test_bucket_name}")
        tests.append(("S3 Write/Delete Access", True, "Can create and delete buckets"))
    except Exception as e:
        logger.error(f"❌ S3 Write Access failed: {str(e)}")
        tests.append(("S3 Write/Delete Access", False, str(e)))
    
    return tests

def create_infrastructure_test():
    """Test infrastructure creation capabilities"""
    
    try:
        # Test CloudFormation access
        cf = boto3.client('cloudformation')
        stacks = cf.list_stacks()
        print(f"✅ CloudFormation Access: Can list stacks")
        return True
    except Exception as e:
        print(f"❌ CloudFormation Access failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("🧪 Testing AWS Connectivity...\n")
    
    test_results = test_aws_connectivity()
    
    print("\n📊 Test Results Summary:")
    print("-" * 50)
    
    passed = 0
    total = len(test_results)
    
    for test_name, success, message in test_results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}: {message}")
        if success:
            passed += 1
    
    print("-" * 50)
    print(f"Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your AWS setup is ready.")
    else:
        print("⚠️ Some tests failed. Please check your AWS configuration.")
    
    # Test infrastructure capabilities
    print("\n🏗️ Testing Infrastructure Capabilities...")
    if create_infrastructure_test():
        print("✅ Infrastructure deployment capabilities confirmed")
    else:
        print("❌ Infrastructure deployment may be limited")
