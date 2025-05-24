import { Controller, Post, Get, Body, UseGuards, Request } from '@nestjs/common';
import { UserService } from './user.service';
import { LoginDto, RegisterDto } from './dto/auth.dto';
import { JwtAuthGuard } from '@shared/guards/jwt-auth.guard';
import { RolesGuard } from '@shared/guards/roles.guard';
import { Roles, UserRole } from '@shared/decorators/roles.decorator';

@Controller('auth')
export class UserController {
  constructor(private userService: UserService) {}

  @Post('register')
  async register(@Body() registerDto: RegisterDto) {
    return this.userService.register(registerDto);
  }

  @Post('login')
  async login(@Body() loginDto: LoginDto) {
    return this.userService.login(loginDto);
  }

  @UseGuards(JwtAuthGuard)
  @Get('profile')
  async getProfile(@Request() req) {
    return req.user;
  }

  @UseGuards(JwtAuthGuard, RolesGuard)
  @Roles(UserRole.TEACHER)
  @Get('teacher-only')
  async teacherOnlyEndpoint(@Request() req) {
    return {
      message: 'This endpoint is only accessible by teachers',
      user: req.user,
    };
  }

  @UseGuards(JwtAuthGuard, RolesGuard)
  @Roles(UserRole.STUDENT)
  @Get('student-only')
  async studentOnlyEndpoint(@Request() req) {
    return {
      message: 'This endpoint is only accessible by students',
      user: req.user,
    };
  }
} 