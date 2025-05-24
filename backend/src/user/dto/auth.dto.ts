import { UserRole } from '@shared/decorators/roles.decorator';

export class LoginDto {
  email: string;
  password: string;
}

export class RegisterDto {
  email: string;
  password: string;
  role: UserRole;
}

export class AuthResponseDto {
  access_token: string;
  user: {
    id: string;
    email: string;
    role: UserRole;
  };
} 