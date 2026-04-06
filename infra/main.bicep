// ── Forecaster Platform — Azure Infrastructure ─────────────────────────
//
// Deploy:
//   az deployment group create -g forecaster-dev-rg  -f main.bicep -p environment=dev  ...
//   az deployment group create -g forecaster-test-rg -f main.bicep -p environment=test ...
//   az deployment group create -g forecaster-prod-rg -f main.bicep -p environment=prod ...
//
// Each environment gets its own resource group and isolated resources.

targetScope = 'resourceGroup'

@description('Environment name')
@allowed(['dev', 'test', 'prod'])
param environment string = 'dev'

@description('Azure region')
param location string = resourceGroup().location

@description('PostgreSQL admin password')
@secure()
param dbAdminPassword string

@description('Application JWT secret')
@secure()
param jwtSecret string

@description('DeepSeek API key')
@secure()
param deepseekApiKey string

// ── Naming ──────────────────────────────────────────────────────────────
var prefix = 'forecaster-${environment}'
var uniqueSuffix = uniqueString(resourceGroup().id)

// ── Environment-specific sizing ─────────────────────────────────────────
var envConfig = {
  dev: {
    pgSku: 'Standard_B1ms'
    pgTier: 'Burstable'
    pgStorageGB: 32
    pgBackupDays: 7
    redisSku: 'Basic'
    redisCapacity: 0
    backendCpu: '0.25'
    backendMemory: '0.5Gi'
    frontendCpu: '0.25'
    frontendMemory: '0.5Gi'
    backendMinReplicas: 0
    backendMaxReplicas: 1
    frontendMinReplicas: 0
    frontendMaxReplicas: 1
    logRetentionDays: 7
    storageRedundancy: 'Standard_LRS'
  }
  test: {
    pgSku: 'Standard_B1ms'
    pgTier: 'Burstable'
    pgStorageGB: 32
    pgBackupDays: 7
    redisSku: 'Basic'
    redisCapacity: 0
    backendCpu: '0.5'
    backendMemory: '1Gi'
    frontendCpu: '0.25'
    frontendMemory: '0.5Gi'
    backendMinReplicas: 0
    backendMaxReplicas: 2
    frontendMinReplicas: 0
    frontendMaxReplicas: 1
    logRetentionDays: 14
    storageRedundancy: 'Standard_LRS'
  }
  prod: {
    pgSku: 'Standard_B2s'
    pgTier: 'Burstable'
    pgStorageGB: 64
    pgBackupDays: 30
    redisSku: 'Basic'
    redisCapacity: 1
    backendCpu: '1'
    backendMemory: '2Gi'
    frontendCpu: '0.5'
    frontendMemory: '1Gi'
    backendMinReplicas: 1
    backendMaxReplicas: 5
    frontendMinReplicas: 1
    frontendMaxReplicas: 3
    logRetentionDays: 90
    storageRedundancy: 'Standard_GRS'
  }
}

var cfg = envConfig[environment]

// ── Container Registry ──────────────────────────────────────────────────
resource acr 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: replace('${prefix}acr${uniqueSuffix}', '-', '')
  location: location
  sku: {
    name: 'Basic'
  }
  properties: {
    adminUserEnabled: true
  }
}

// ── PostgreSQL Flexible Server ──────────────────────────────────────────
resource pgServer 'Microsoft.DBforPostgreSQL/flexibleServers@2023-06-01-preview' = {
  name: '${prefix}-pg-${uniqueSuffix}'
  location: location
  sku: {
    name: cfg.pgSku
    tier: cfg.pgTier
  }
  properties: {
    version: '16'
    administratorLogin: 'forecaster_admin'
    administratorLoginPassword: dbAdminPassword
    storage: {
      storageSizeGB: cfg.pgStorageGB
    }
    backup: {
      backupRetentionDays: cfg.pgBackupDays
      geoRedundantBackup: environment == 'prod' ? 'Enabled' : 'Disabled'
    }
    highAvailability: {
      mode: 'Disabled'
    }
  }
}

// Allow Azure services to connect
resource pgFirewall 'Microsoft.DBforPostgreSQL/flexibleServers/firewallRules@2023-06-01-preview' = {
  parent: pgServer
  name: 'AllowAzureServices'
  properties: {
    startIpAddress: '0.0.0.0'
    endIpAddress: '0.0.0.0'
  }
}

// Database
resource pgDatabase 'Microsoft.DBforPostgreSQL/flexibleServers/databases@2023-06-01-preview' = {
  parent: pgServer
  name: 'forecaster'
  properties: {
    charset: 'UTF8'
    collation: 'en_US.utf8'
  }
}

// ── Storage Account (Blob) ──────────────────────────────────────────────
resource storageAccount 'Microsoft.Storage/storageAccounts@2023-01-01' = {
  name: replace('${prefix}st${uniqueSuffix}', '-', '')
  location: location
  kind: 'StorageV2'
  sku: {
    name: cfg.storageRedundancy
  }
  properties: {
    accessTier: 'Hot'
    supportsHttpsTrafficOnly: true
    minimumTlsVersion: 'TLS1_2'
  }
}

// Blob containers
resource blobService 'Microsoft.Storage/storageAccounts/blobServices@2023-01-01' = {
  parent: storageAccount
  name: 'default'
}

resource datasetsContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-01-01' = {
  parent: blobService
  name: 'datasets'
}

resource modelsContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-01-01' = {
  parent: blobService
  name: 'models'
}

resource forecastsContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-01-01' = {
  parent: blobService
  name: 'forecasts'
}

// ── Redis Cache ─────────────────────────────────────────────────────────
resource redis 'Microsoft.Cache/redis@2023-08-01' = {
  name: '${prefix}-redis-${uniqueSuffix}'
  location: location
  properties: {
    sku: {
      name: cfg.redisSku
      family: 'C'
      capacity: cfg.redisCapacity
    }
    enableNonSslPort: false
    minimumTlsVersion: '1.2'
  }
}

// ── Log Analytics ───────────────────────────────────────────────────────
resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2022-10-01' = {
  name: '${prefix}-logs-${uniqueSuffix}'
  location: location
  properties: {
    sku: {
      name: 'PerGB2018'
    }
    retentionInDays: cfg.logRetentionDays
  }
}

// ── Container Apps Environment ──────────────────────────────────────────
resource containerEnv 'Microsoft.App/managedEnvironments@2023-05-01' = {
  name: '${prefix}-env'
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logAnalytics.properties.customerId
        sharedKey: logAnalytics.listKeys().primarySharedKey
      }
    }
  }
}

// ── Backend Container App ───────────────────────────────────────────────
resource backendApp 'Microsoft.App/containerApps@2023-05-01' = {
  name: '${prefix}-backend'
  location: location
  properties: {
    managedEnvironmentId: containerEnv.id
    configuration: {
      ingress: {
        external: true
        targetPort: 8000
        transport: 'auto'
        corsPolicy: {
          allowedOrigins: ['*']
          allowedMethods: ['GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'OPTIONS']
          allowedHeaders: ['*']
        }
      }
      registries: [
        {
          server: acr.properties.loginServer
          username: acr.listCredentials().username
          passwordSecretRef: 'acr-password'
        }
      ]
      secrets: [
        {
          name: 'acr-password'
          value: acr.listCredentials().passwords[0].value
        }
        {
          name: 'database-url'
          value: 'postgresql+asyncpg://forecaster_admin:${dbAdminPassword}@${pgServer.properties.fullyQualifiedDomainName}:5432/forecaster?ssl=require'
        }
        {
          name: 'redis-url'
          value: 'rediss://:${redis.listKeys().primaryKey}@${redis.properties.hostName}:6380/0'
        }
        {
          name: 'jwt-secret'
          value: jwtSecret
        }
        {
          name: 'deepseek-key'
          value: deepseekApiKey
        }
        {
          name: 'storage-connection'
          value: 'DefaultEndpointsProtocol=https;AccountName=${storageAccount.name};AccountKey=${storageAccount.listKeys().keys[0].value};EndpointSuffix=core.windows.net'
        }
      ]
    }
    template: {
      containers: [
        {
          name: 'backend'
          image: '${acr.properties.loginServer}/forecaster-backend:latest'
          resources: {
            cpu: json(cfg.backendCpu)
            memory: cfg.backendMemory
          }
          env: [
            { name: 'DATABASE_URL', secretRef: 'database-url' }
            { name: 'REDIS_URL', secretRef: 'redis-url' }
            { name: 'CELERY_BROKER_URL', secretRef: 'redis-url' }
            { name: 'CELERY_RESULT_BACKEND', secretRef: 'redis-url' }
            { name: 'SECRET_KEY', secretRef: 'jwt-secret' }
            { name: 'DEEPSEEK_API_KEY', secretRef: 'deepseek-key' }
            { name: 'AZURE_STORAGE_CONNECTION_STRING', secretRef: 'storage-connection' }
            { name: 'ENVIRONMENT', value: environment }
          ]
        }
      ]
      scale: {
        minReplicas: cfg.backendMinReplicas
        maxReplicas: cfg.backendMaxReplicas
        rules: [
          {
            name: 'http-rule'
            http: {
              metadata: {
                concurrentRequests: '50'
              }
            }
          }
        ]
      }
    }
  }
}

// ── Frontend Container App ──────────────────────────────────────────────
resource frontendApp 'Microsoft.App/containerApps@2023-05-01' = {
  name: '${prefix}-frontend'
  location: location
  properties: {
    managedEnvironmentId: containerEnv.id
    configuration: {
      ingress: {
        external: true
        targetPort: 3000
        transport: 'auto'
      }
      registries: [
        {
          server: acr.properties.loginServer
          username: acr.listCredentials().username
          passwordSecretRef: 'acr-password'
        }
      ]
      secrets: [
        {
          name: 'acr-password'
          value: acr.listCredentials().passwords[0].value
        }
      ]
    }
    template: {
      containers: [
        {
          name: 'frontend'
          image: '${acr.properties.loginServer}/forecaster-frontend:latest'
          resources: {
            cpu: json(cfg.frontendCpu)
            memory: cfg.frontendMemory
          }
          env: [
            {
              name: 'NEXT_PUBLIC_API_URL'
              value: 'https://${backendApp.properties.configuration.ingress.fqdn}'
            }
            {
              name: 'ENVIRONMENT'
              value: environment
            }
          ]
        }
      ]
      scale: {
        minReplicas: cfg.frontendMinReplicas
        maxReplicas: cfg.frontendMaxReplicas
      }
    }
  }
}

// ── Celery Worker Container App ─────────────────────────────────────────
resource celeryApp 'Microsoft.App/containerApps@2023-05-01' = {
  name: '${prefix}-worker'
  location: location
  properties: {
    managedEnvironmentId: containerEnv.id
    configuration: {
      registries: [
        {
          server: acr.properties.loginServer
          username: acr.listCredentials().username
          passwordSecretRef: 'acr-password'
        }
      ]
      secrets: [
        {
          name: 'acr-password'
          value: acr.listCredentials().passwords[0].value
        }
        {
          name: 'database-url'
          value: 'postgresql+asyncpg://forecaster_admin:${dbAdminPassword}@${pgServer.properties.fullyQualifiedDomainName}:5432/forecaster?ssl=require'
        }
        {
          name: 'redis-url'
          value: 'rediss://:${redis.listKeys().primaryKey}@${redis.properties.hostName}:6380/0'
        }
        {
          name: 'deepseek-key'
          value: deepseekApiKey
        }
        {
          name: 'storage-connection'
          value: 'DefaultEndpointsProtocol=https;AccountName=${storageAccount.name};AccountKey=${storageAccount.listKeys().keys[0].value};EndpointSuffix=core.windows.net'
        }
      ]
    }
    template: {
      containers: [
        {
          name: 'worker'
          image: '${acr.properties.loginServer}/forecaster-backend:latest'
          resources: {
            cpu: json(cfg.backendCpu)
            memory: cfg.backendMemory
          }
          command: [
            'celery'
            '-A'
            'app.tasks'
            'worker'
            '--loglevel=info'
            '--concurrency=2'
            '-Q'
            'training,forecast,etl'
          ]
          env: [
            { name: 'DATABASE_URL', secretRef: 'database-url' }
            { name: 'REDIS_URL', secretRef: 'redis-url' }
            { name: 'CELERY_BROKER_URL', secretRef: 'redis-url' }
            { name: 'CELERY_RESULT_BACKEND', secretRef: 'redis-url' }
            { name: 'DEEPSEEK_API_KEY', secretRef: 'deepseek-key' }
            { name: 'AZURE_STORAGE_CONNECTION_STRING', secretRef: 'storage-connection' }
            { name: 'ENVIRONMENT', value: environment }
          ]
        }
      ]
      scale: {
        minReplicas: environment == 'prod' ? 1 : 0
        maxReplicas: environment == 'prod' ? 3 : 1
        rules: [
          {
            name: 'cpu-rule'
            custom: {
              type: 'cpu'
              metadata: {
                type: 'Utilization'
                value: '70'
              }
            }
          }
        ]
      }
    }
  }
}

// ── Outputs ─────────────────────────────────────────────────────────────
output acrLoginServer string = acr.properties.loginServer
output backendUrl string = 'https://${backendApp.properties.configuration.ingress.fqdn}'
output frontendUrl string = 'https://${frontendApp.properties.configuration.ingress.fqdn}'
output pgHost string = pgServer.properties.fullyQualifiedDomainName
output storageAccountName string = storageAccount.name
