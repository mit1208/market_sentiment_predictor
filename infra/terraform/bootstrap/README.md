# One-Time Project Bootstrap

This Terraform stack creates a brand new GCP project, attaches billing, enables the base APIs
needed for the rest of the infrastructure, and can optionally grant your friend immediate access.

## Usage

1. Initialize the bootstrap stack:
   ```bash
   cd infra/terraform/bootstrap
   cp terraform.tfvars.example terraform.tfvars
   terraform init
   ```

2. Fill in `terraform.tfvars`:
   - `project_id`: globally unique project ID
   - `project_name`: display name shown in the console
   - `billing_account_id`: billing account to attach
   - `folder_id` or `organization_id`: optional parent placement
   - `friend_user_email`: optional account to add right away

3. Apply the bootstrap:
   ```bash
   terraform apply
   ```

4. Deploy the application stack into the new project:
   ```bash
   cd ../
   cp terraform.tfvars.example terraform.tfvars
   ```

   Then set `project_id` to the new project and apply the main stack from `infra/terraform`.

The bootstrap stack is intentionally separate from the application stack so project creation stays a one-time operation and routine app deploys do not manage project lifecycle.
